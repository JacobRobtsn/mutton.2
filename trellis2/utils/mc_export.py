"""
Convert TRELLIS.2 MeshWithVoxel output to Minecraft .schem (Sponge Schematic v2).

GPU-accelerated pipeline:
  1. Downsample sparse voxels on GPU (torch unique + scatter)
  2. Color-to-block matching on GPU (brute-force L2, ~150 palette entries)
  3. Scatter block indices into flat volume on GPU
  4. Transfer only the final int32 array to CPU for varint + NBT serialization
"""
import io
import gzip
import struct
from typing import Optional
import numpy as np
import torch

from .mc_palette import match_colors_gpu, BLOCK_NAMES


# ──────────────────────────────────────────────
# Minimal NBT writer
# ──────────────────────────────────────────────

_TAG_END = 0
_TAG_SHORT = 2
_TAG_INT = 3
_TAG_BYTE_ARRAY = 7
_TAG_STRING = 8
_TAG_LIST = 9
_TAG_COMPOUND = 10
_TAG_INT_ARRAY = 11


def _whdr(buf, tt, name):
    buf.write(struct.pack(">b", tt))
    e = name.encode("utf-8")
    buf.write(struct.pack(">H", len(e)))
    buf.write(e)

def _wshort(buf, n, v):  _whdr(buf, _TAG_SHORT, n);   buf.write(struct.pack(">h", v))
def _wint(buf, n, v):    _whdr(buf, _TAG_INT, n);      buf.write(struct.pack(">i", v))
def _wstr(buf, n, v):    _whdr(buf, _TAG_STRING, n);   e=v.encode("utf-8"); buf.write(struct.pack(">H", len(e))); buf.write(e)
def _wbarr(buf, n, d):   _whdr(buf, _TAG_BYTE_ARRAY, n); buf.write(struct.pack(">i", len(d))); buf.write(d)
def _wiarr(buf, n, vs):  _whdr(buf, _TAG_INT_ARRAY, n); buf.write(struct.pack(">i", len(vs))); buf.write(struct.pack(f">{len(vs)}i", *vs))
def _wcstart(buf, n):    _whdr(buf, _TAG_COMPOUND, n)
def _wcend(buf):          buf.write(b'\x00')
def _wlist_empty(buf, n, elem_type):
    """Write an empty TAG_List."""
    _whdr(buf, _TAG_LIST, n)
    buf.write(struct.pack(">b", elem_type))   # element tag type
    buf.write(struct.pack(">i", 0))            # length = 0


# ──────────────────────────────────────────────
# Vectorized varint encoder (numpy, no Python loop)
# ──────────────────────────────────────────────

def _varint_encode_batch(values: np.ndarray) -> bytes:
    """
    Encode an array of non-negative int32s as concatenated varints.
    Palette sizes are small (<256 typically), so most values fit in 1-2 bytes.
    Vectorized: splits into byte-lanes then masks, avoids per-element Python loop.
    """
    v = values.astype(np.uint32)
    # Max varint length for values < 2^28 is 4 bytes; palette indices are tiny
    b0 = (v & 0x7F).astype(np.uint8)
    v1 = v >> 7
    b1 = (v1 & 0x7F).astype(np.uint8)
    v2 = v1 >> 7
    b2 = (v2 & 0x7F).astype(np.uint8)
    v3 = v2 >> 7
    b3 = (v3 & 0x7F).astype(np.uint8)

    need1 = v1 > 0   # needs at least 2 bytes
    need2 = v2 > 0   # needs at least 3 bytes
    need3 = v3 > 0   # needs at least 4 bytes

    # Set continuation bits
    b0[need1] |= 0x80
    b1[need2] |= 0x80
    b2[need3] |= 0x80

    # Count total bytes needed
    lengths = np.ones(len(values), dtype=np.int32)
    lengths += need1.astype(np.int32)
    lengths += need2.astype(np.int32)
    lengths += need3.astype(np.int32)
    total = int(lengths.sum())

    # Build output
    out = np.empty(total, dtype=np.uint8)
    offsets = np.empty(len(values) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])

    # Scatter bytes (vectorized via fancy indexing)
    out[offsets[:-1]] = b0
    mask1 = need1
    out[offsets[:-1][mask1] + 1] = b1[mask1]
    mask2 = need2
    out[offsets[:-1][mask2] + 2] = b2[mask2]
    mask3 = need3
    out[offsets[:-1][mask3] + 3] = b3[mask3]

    return out.tobytes()


# ──────────────────────────────────────────────
# GPU downsample
# ──────────────────────────────────────────────

@torch.no_grad()
def _downsample_gpu(coords: torch.Tensor, colors: torch.Tensor,
                    native_res: int, target_res: int):
    """
    Downsample sparse voxels on GPU by binning + averaging colors.
    
    Args:
        coords: (N, 3) int tensor on CUDA
        colors: (N, 3) float tensor on CUDA, [0, 1]
        native_res: original grid size
        target_res: desired grid size
    
    Returns:
        new_coords (M, 3) int, new_colors (M, 3) float — both on CUDA
    """
    if target_res >= native_res:
        return coords, colors

    ratio = native_res / target_res
    binned = (coords.float() / ratio).int().clamp(0, target_res - 1)

    # 1D keys for unique
    tr2 = target_res * target_res
    keys = binned[:, 0] * tr2 + binned[:, 1] * target_res + binned[:, 2]

    unique_keys, inverse = torch.unique(keys, return_inverse=True)
    M = unique_keys.shape[0]

    # Scatter-add colors and counts
    new_colors = torch.zeros(M, 3, device=colors.device, dtype=torch.float32)
    counts = torch.zeros(M, 1, device=colors.device, dtype=torch.float32)
    new_colors.scatter_add_(0, inverse.unsqueeze(1).expand(-1, 3), colors)
    counts.scatter_add_(0, inverse.unsqueeze(1), torch.ones(keys.shape[0], 1, device=colors.device))
    new_colors /= counts

    # Decode keys
    new_coords = torch.stack([
        unique_keys // tr2,
        (unique_keys // target_res) % target_res,
        unique_keys % target_res,
    ], dim=1).int()

    return new_coords, new_colors


# ──────────────────────────────────────────────
# .schem builder (CPU serialization of GPU results)
# ──────────────────────────────────────────────

def _build_schem_bytes(coords_cpu: np.ndarray, palette_idxs_cpu: np.ndarray) -> bytes:
    """
    Build gzipped Sponge Schematic v2 from zero-origin coords + per-voxel palette indices.
    coords_cpu: (N, 3) int32
    palette_idxs_cpu: (N,) int32, indices into BLOCK_NAMES (+ offset by 1 for air at 0)
    """
    # Shift to zero-origin
    min_c = coords_cpu.min(axis=0)
    coords_cpu = coords_cpu - min_c
    max_c = coords_cpu.max(axis=0)

    width  = int(max_c[0]) + 1
    height = int(max_c[1]) + 1
    length = int(max_c[2]) + 1

    # Collect unique palette entries actually used
    used_palette = np.unique(palette_idxs_cpu)
    # Build compact palette: air=0, then each used block
    block_names = ["minecraft:air"]
    remap = {}  # old palette idx -> new compact idx
    for old_idx in used_palette:
        remap[int(old_idx)] = len(block_names)
        block_names.append(BLOCK_NAMES[int(old_idx)])

    # Fill flat volume (default 0 = air)
    total = width * height * length
    flat = np.zeros(total, dtype=np.int32)

    # Vectorized scatter: compute linear indices then assign
    x = coords_cpu[:, 0].astype(np.int64)
    y = coords_cpu[:, 1].astype(np.int64)
    z = coords_cpu[:, 2].astype(np.int64)
    lin = y * length * width + z * width + x
    # Remap palette indices
    remap_arr = np.zeros(len(BLOCK_NAMES), dtype=np.int32)
    for old_idx, new_idx in remap.items():
        remap_arr[old_idx] = new_idx
    flat[lin] = remap_arr[palette_idxs_cpu]

    # Varint-encode block data
    block_data = _varint_encode_batch(flat)

    # Build palette map for NBT
    palette_map = {name: i for i, name in enumerate(block_names)}

    # Write NBT — Sponge Schematic v2
    # Root is an unnamed compound wrapping "Schematic" compound for max compat
    buf = io.BytesIO()
    _wcstart(buf, "Schematic")
    _wint(buf, "Version", 2)
    _wint(buf, "DataVersion", 3578)
    _wshort(buf, "Width", width)
    _wshort(buf, "Height", height)
    _wshort(buf, "Length", length)

    _wcstart(buf, "Palette")
    for name, idx in palette_map.items():
        _wint(buf, name, idx)
    _wcend(buf)

    _wint(buf, "PaletteMax", len(palette_map))
    _wbarr(buf, "BlockData", block_data)

    # BlockEntities — required by WorldEdit/FAWE even if empty
    _wlist_empty(buf, "BlockEntities", _TAG_COMPOUND)

    _wcstart(buf, "Metadata")
    _wstr(buf, "Generator", "TRELLIS2-Minecraft")
    _wcend(buf)

    _wiarr(buf, "Offset", [0, 0, 0])
    _wcend(buf)

    raw = buf.getvalue()
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb") as gz:
        gz.write(raw)
    return out.getvalue()


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

_MAX_VOLUME = 8_000_000  # 200^3 — safe upper bound for .schem viewers

@torch.no_grad()
def to_schem(
    mesh_with_voxel,
    target_resolution: Optional[int] = 128,
    output_path: Optional[str] = None,
) -> bytes:
    """
    Convert a TRELLIS.2 MeshWithVoxel to a Minecraft .schem file.
    
    GPU-accelerated: downsample + color matching run on CUDA.
    Only the final NBT serialization happens on CPU.
    
    Args:
        mesh_with_voxel: MeshWithVoxel from pipeline output
        target_resolution: downsample to this block count per axis (default 128).
        output_path: if set, write .schem to this path
    
    Returns:
        gzipped .schem bytes
    """
    m = mesh_with_voxel
    device = m.coords.device

    # ── 1. Extract voxel data (stay on GPU) ──
    coords = m.coords.int()                                  # (N, 3)
    color_slice = m.layout.get("base_color", slice(0, 3))
    colors = m.attrs[:, color_slice].float()                  # (N, 3) in [0,1]

    native_res = int(coords.max().item()) + 1

    # ── 2. GPU downsample (always downsample if native > target) ──
    if target_resolution is not None and target_resolution < native_res:
        coords, colors = _downsample_gpu(coords, colors, native_res, target_resolution)
    elif native_res > 256:
        # Safety cap: even if no target given, don't exceed 256
        coords, colors = _downsample_gpu(coords, colors, native_res, 256)

    # ── 3. GPU color matching ──
    colors_255 = (colors * 255.0).clamp(0, 255)
    palette_idxs = match_colors_gpu(colors_255)               # (M,) int64 on GPU

    # ── 4. Single transfer to CPU ──
    coords_cpu = coords.cpu().numpy().astype(np.int32)
    palette_idxs_cpu = palette_idxs.cpu().numpy().astype(np.int32)

    # ── 5. Serialize (CPU) ──
    schem_bytes = _build_schem_bytes(coords_cpu, palette_idxs_cpu)

    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(schem_bytes)

    return schem_bytes
