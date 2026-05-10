# ColabFold patches — pan-barley orphan protein folding

This branch (`gspmd-dual-gpu-perf`) extends ColabFold 1.6.1 to fold very large
proteins (3800–5400 aa) that exceed the memory of a single 40 GB A100. It builds
on a throughput-optimisation layer and adds JAX GSPMD tensor sharding across two
GPUs, plus a critical subbatch-size fix that reduces TriangleAttention kernel launch
overhead by ~60×.

Tested hardware: 2× NVIDIA A100-SXM4-40GB (NVLink), SLURM `compregular` partition.

---

## Patches

### 1. `colabfold/batch.py` — async I/O writer

**Problem**: PDB and score-JSON writes to NFS block the GPU dispatch loop.
For large proteins (>3000 aa) the pair-representation array serialisation
(`np.save` on a 3000²×128 float32 array ≈ 4 GB) stalls the GPU for seconds.

**Fix**: A `ThreadPoolExecutor(max_workers=3)` offloads all file writes from
the main thread. Arrays are `.copy()`-ed before submission so the GPU can
immediately reclaim the JAX buffer.

**Rationale**: NFS latency is unpredictable (1 ms – 10 s on our cluster).
Decoupling writes from the GPU loop prevents a single slow mount from
serialising multiple recycles.

---

### 2. `colabfold/batch.py` — multiplicative recompile padding

**Problem**: Sequence length is padded to the next multiple of `recompile_padding`
(default 10) for XLA cache reuse. For proteins in the 3800–5400 aa range this
triggers many distinct compilations.

**Fix**: Padding is computed as `ceil(L / recompile_padding) * recompile_padding`
where `recompile_padding` grows multiplicatively (powers of 2 bucketing). Reduces
distinct XLA compilations for the length range of interest from O(L) to O(log L).

---

### 3. `colabfold/batch.py` — PAE numpy serialisation

**Problem**: `pae.tolist()` in the score-JSON path materialises O(N²) Python
float objects — at N=4000 this is 16 million object allocations on the main thread,
adding ~2 s per recycle.

**Fix**: Use `orjson.OPT_SERIALIZE_NUMPY` to serialise the raw NumPy float32 array
directly; fall back to `.tolist()` only when orjson is unavailable.

---

### 4. `colabfold/batch.py` — single params load

**Problem**: Model parameters (~700 MB per model) were loaded once per recycle
in some code paths.

**Fix**: Parameters are loaded once per `(model_name, random_seed)` pair and
reused across recycles. Combined with the async writer, GPU utilisation on the
hot recycle path increases from ~60% to ~90%.

---

### 5. `colabfold/alphafold/models.py` — `num_gpus` parameter

**Problem**: No mechanism to pass a multi-GPU count to `RunModel`.

**Fix**: Adds `num_gpus: int = 1` to `load_models_and_params` and threads it
through to `RunModel.__init__`. When `num_gpus > 1`, `RunModel` sets up a JAX
GSPMD mesh (see alphafold patch below).

---

### 6. `colabfold/batch.py` — `--num-gpus` CLI flag

**Problem**: `colabfold_batch` has no way to request multi-GPU folding.

**Fix**: Adds `--num-gpus N` argument (default 1). The value is forwarded to
`load_models_and_params` → `RunModel.__init__`. When N > 1, JAX GSPMD sharding
is activated across N devices. Requires setting `--gres=gpu:N` in SLURM.

---

### 7. `alphafold/model/model.py` — JAX GSPMD mesh + correct sharding axis

**Problem**: The pair representation `(N_res, N_res, c_z=128)` is ~5 GB at
N=4032, exceeding a single 40 GB A100 when combined with MSA activations
and intermediate buffers.

**Fix** (`patches/alphafold/model.py.patch`):

```python
# When num_gpus > 1:
devices = mesh_utils.create_device_mesh((num_gpus,))
self._mesh = Mesh(devices, ('seq',))
modules._PAIR_SHARDING = NamedSharding(self._mesh, P(None, None, 'seq'))
modules._MSA_SHARDING  = NamedSharding(self._mesh, P(None, 'seq', None))
```

**Critical axis choice**: `P(None, None, 'seq')` shards on the *channel* dimension
(`c_z=128`), not on the residue rows (`P('seq', None, None)`).

Why this matters: `TriangleAttentionEndingNode` transposes the pair tensor
`(i, j, c)` → `(j, i, c)` before computing attention along axis 0. Sharding on
axis 0 (residue rows) would require an `all-gather` inside every triangle-attention
call (inside `lax.scan`), making multi-GPU slower than single-GPU due to ~1000
all-gather collectives per recycle. Sharding on the channel dim (axis 2) propagates
through the transpose without triggering any collective — XLA only needs one
`all-gather` at the very end of the Evoformer to reconstruct the full tensor.

---

### 8. `alphafold/model/modules.py` — GSPMD sharding insertion points

**Fix** (`patches/alphafold/modules.py.patch`):

```python
# Global module-level sharding descriptors (set by model.py when num_gpus > 1)
_PAIR_SHARDING = None
_MSA_SHARDING  = None

def _maybe_shard_pair(x): ...
def _maybe_shard_msa(x):  ...
```

Sharding constraints are inserted at **three** points:

1. **Before Extra-MSA stack** (line ~1975): pair and extra-MSA activations are
   sharded once before entering the 4-block Extra-MSA stack. This ensures XLA
   assigns the correct device placement without an all-gather at the stack boundary.

2. **After Extra-MSA stack** (line ~2007): re-shard pair activations after the
   stack to anchor placement before the Evoformer input construction.

3. **Inside Evoformer `lax.scan` body** (lines 2077–2078): originally inserted
   `_maybe_shard_pair` / `_maybe_shard_msa` per iteration. **Removed** because
   `lax.scan` propagates sharding from the initial carry automatically — inserting
   a `with_sharding_constraint` every iteration adds ~96 redundant collective
   annotations (48 blocks × 2) and confuses the XLA GSPMD placer into inserting
   unnecessary all-gathers.

---

### 9. `alphafold/model/config.py` — subbatch_size 4 → 256

**Problem**: `inference_subbatch` (the `subbatch_size` parameter) controls how
many residue rows are processed per iteration in `TriangleAttention` via `hk.scan`.
The default value is **4**.

At N=4032 aa:
- Iterations per TriangleAttention call: `ceil(4032/4)` = 1008
- TriangleAttention calls per Evoformer block: 2 (starting + ending node)
- Evoformer blocks: 48
- Total `hk.scan` iterations: 1008 × 2 × 48 ≈ **96 768 per recycle**

Each iteration is a tiny GPU kernel launch. The XLA CUDA executor spends more
time dispatching than computing at this granularity. At 3 recycles, the 11 OOM
proteins accumulate ~3 million kernel launches, explaining why wall time exceeded
24 h on a single GPU and ~12 h/recycle even with two.

**Fix** (`patches/alphafold/config.py.patch`): `subbatch_size`: 4 → **256**

At N=4032: `ceil(4032/256)` = 16 iterations → **~61× fewer kernel launches**.
Expected wall-time reduction: ~10× for the TriangleAttention-dominated path.

Changed for both monomer (`multimer_mode: False`) and multimer (`multimer_mode: True`)
configs. The value 256 fits comfortably in 40 GB even without GPU sharding: the
subbatch pair slice is `256 × N_res × c_z` = `256 × 4032 × 128` ≈ 500 MB.

---

## Environment variables

Add to SLURM job scripts:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_FLAGS="--xla_gpu_enable_async_collectives=true --xla_gpu_enable_triton_gemm=true"
export PYTHONPATH=/path/to/colabfold-patched:$PYTHONPATH
```

`XLA_PYTHON_CLIENT_PREALLOCATE=false`: prevents JAX from pre-allocating 90% of GPU
memory at startup. Without this, the XLA kernel loader fails with
`CUDA_ERROR_OUT_OF_MEMORY: Failed to get module function` because the pre-allocated
arena blocks CUDA from mapping compiled kernel code.

`XLA_GPU_ENABLE_ASYNC_COLLECTIVES`: overlaps NCCL all-gather/reduce-scatter
collectives with compute on the NVLink bus. ~5–15% throughput gain for the
GSPMD sharding collectives.

`XLA_GPU_ENABLE_TRITON_GEMM`: routes GEMMs ≥ a threshold size through OpenAI Triton
kernels instead of cuBLAS. Gives ~10–20% speedup for the large attention projection
GEMMs at N>2000.

---

## Installation

```bash
# 1. Install ColabFold normally (follow upstream README)

# 2. Apply alphafold model patches
bash patches/apply_alphafold_patches.sh /path/to/colabfold-env/bin/python3

# 3. Use this repo as the PYTHONPATH overlay
export PYTHONPATH=/path/to/colabfold-patched:$PYTHONPATH
```

## Usage

```bash
colabfold_batch \
    --model-type alphafold2_ptm \
    --num-models 1 \
    --num-recycle 3 \
    --num-gpus 2 \
    input.a3m output_dir/
```

`--num-gpus 1` activates no sharding (standard single-GPU path).
`--num-gpus 2` activates GSPMD across 2× A100s.
