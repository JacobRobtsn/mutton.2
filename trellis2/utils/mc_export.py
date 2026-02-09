"""
Convert TRELLIS.2 MeshWithVoxel output to Minecraft .schem (Sponge Schematic v2).

GPU-accelerated pipeline:
  1. Downsample sparse voxels on GPU (torch unique + scatter)
  2. Color-to-block matching on GPU (brute-force L2, ~150 palette entries)
  3. Geometry analysis on GPU (neighbor masks → slab/stair/fence/wall classification)
  4. Resolve per-voxel block state strings (CPU)
  5. Serialize NBT + gzip (CPU)
"""
import io
import gzip
import struct
from typing import Optional
import numpy as np
import torch

from .mc_palette import (
    match_colors_gpu, BLOCK_NAMES, MATERIAL_FAMILIES, build_block_state,
)
from .mc_geometry import (
    classify_and_resolve,
    FULL_BLOCK, BOTTOM_SLAB, TOP_SLAB, STAIR, FENCE,
    DOOR_LOWER, DOOR_UPPER, TRAPDOOR,
    _EAST, _WEST, _SOUTH, _NORTH, _FACING_STR,
)


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
# Block state resolution (CPU)
# ──────────────────────────────────────────────

_FACING_MAP = {int(_EAST): "east", int(_WEST): "west",
               int(_SOUTH): "south", int(_NORTH): "north"}


def _resolve_block_states(palette_idxs, classifications, facing, half, horiz_neighbors):
    """
    Resolve each voxel to a full block state string combining color-match
    palette index, geometry classification, and directional properties.

    Returns list[str] of length N.
    """
    # Start with all full blocks (fast numpy lookup)
    block_names_arr = np.array(BLOCK_NAMES)
    result = block_names_arr[palette_idxs].tolist()

    # Only loop over special (non-full-block) voxels
    special = np.where(classifications != FULL_BLOCK)[0]

    for i in special:
        base = result[i]
        family = MATERIAL_FAMILIES.get(base)
        if family is None:
            continue  # no variants → stays as full block

        c = int(classifications[i])

        if c in (BOTTOM_SLAB, TOP_SLAB):
            if "slab" not in family:
                continue
            stype = "bottom" if c == BOTTOM_SLAB else "top"
            result[i] = build_block_state(family["slab"],
                {"type": stype, "waterlogged": "false"})

        elif c == STAIR:
            if "stairs" not in family:
                continue
            fd = _FACING_MAP.get(int(facing[i]), "north")
            h  = "bottom" if half[i] == 0 else "top"
            result[i] = build_block_state(family["stairs"],
                {"facing": fd, "half": h, "shape": "straight",
                 "waterlogged": "false"})

        elif c == FENCE:
            if "fence" in family:
                result[i] = build_block_state(family["fence"], {
                    "east":  "true" if horiz_neighbors[i, 0] else "false",
                    "north": "true" if horiz_neighbors[i, 3] else "false",
                    "south": "true" if horiz_neighbors[i, 2] else "false",
                    "west":  "true" if horiz_neighbors[i, 1] else "false",
                    "waterlogged": "false",
                })
            elif "wall" in family:
                result[i] = build_block_state(family["wall"], {
                    "east":  "low" if horiz_neighbors[i, 0] else "none",
                    "north": "low" if horiz_neighbors[i, 3] else "none",
                    "south": "low" if horiz_neighbors[i, 2] else "none",
                    "west":  "low" if horiz_neighbors[i, 1] else "none",
                    "up": "true", "waterlogged": "false",
                })

        elif c in (DOOR_LOWER, DOOR_UPPER):
            if "door" not in family:
                continue
            fd = _FACING_MAP.get(int(facing[i]), "north")
            dh = "lower" if c == DOOR_LOWER else "upper"
            result[i] = build_block_state(family["door"], {
                "facing": fd, "half": dh, "hinge": "left",
                "open": "false", "powered": "false",
            })

        elif c == TRAPDOOR:
            if "trapdoor" not in family:
                continue
            fd = _FACING_MAP.get(int(facing[i]), "north")
            result[i] = build_block_state(family["trapdoor"], {
                "facing": fd, "half": "bottom", "open": "false",
                "powered": "false", "waterlogged": "false",
            })

    return result


# ──────────────────────────────────────────────
# .schem builder (CPU serialization)
# ──────────────────────────────────────────────

def _build_schem_bytes(coords_cpu: np.ndarray, block_states: list) -> bytes:
    """
    Build gzipped Sponge Schematic v2 from coords + per-voxel block state strings.
    coords_cpu:    (N, 3) int32
    block_states:  list[str] length N
    """
    # Shift to zero-origin
    min_c = coords_cpu.min(axis=0)
    coords_cpu = coords_cpu - min_c
    max_c = coords_cpu.max(axis=0)

    # TRELLIS coords are (X, Z, Y) — swap col 1↔2 for Minecraft Y-up
    width  = int(max_c[0]) + 1   # X
    height = int(max_c[2]) + 1   # TRELLIS Z → MC Y (up)
    length = int(max_c[1]) + 1   # TRELLIS Y → MC Z (depth)

    # Build palette from unique block state strings
    states_arr = np.array(block_states, dtype=object)
    unique_states, inverse = np.unique(states_arr, return_inverse=True)

    palette_map = {"minecraft:air": 0}
    for s in unique_states:
        s = str(s)
        if s not in palette_map:
            palette_map[s] = len(palette_map)

    # Map each voxel to compact palette index
    state_idx_map = np.array([palette_map[str(s)] for s in unique_states], dtype=np.int32)
    voxel_palette = state_idx_map[inverse]

    # Fill flat volume (default 0 = air)
    total = width * height * length
    flat = np.zeros(total, dtype=np.int32)

    # Vectorized scatter
    x = coords_cpu[:, 0].astype(np.int64)
    y = coords_cpu[:, 2].astype(np.int64)   # TRELLIS Z → MC Y
    z = coords_cpu[:, 1].astype(np.int64)   # TRELLIS Y → MC Z
    lin = y * length * width + z * width + x
    flat[lin] = voxel_palette

    # Varint-encode block data
    block_data = _varint_encode_batch(flat)

    # Write NBT — Sponge Schematic v2
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
    detect_doors: bool = False,
) -> bytes:
    """
    Convert a TRELLIS.2 MeshWithVoxel to a Minecraft .schem file.
    
    GPU-accelerated: downsample + color matching + geometry analysis on CUDA.
    Only block-state resolution and NBT serialization happen on CPU.
    
    Args:
        mesh_with_voxel: MeshWithVoxel from pipeline output
        target_resolution: downsample to this block count per axis (default 128).
        output_path: if set, write .schem to this path
        detect_doors: enable door/trapdoor detection (prone to false positives)
    
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

    # ── 2. GPU downsample ──
    if target_resolution is not None and target_resolution < native_res:
        coords, colors = _downsample_gpu(coords, colors, native_res, target_resolution)
    elif native_res > 256:
        coords, colors = _downsample_gpu(coords, colors, native_res, 256)

    # ── 3. GPU color matching ──
    colors_255 = (colors * 255.0).clamp(0, 255)
    palette_idxs = match_colors_gpu(colors_255)               # (M,) int64 on GPU

    # ── 4. GPU geometry analysis ──
    grid_size = int(coords.max().item()) + 1
    classifications, geo_props = classify_and_resolve(
        coords, grid_size, detect_doors=detect_doors)

    # ── 5. Single transfer to CPU ──
    coords_cpu      = coords.cpu().numpy().astype(np.int32)
    palette_cpu     = palette_idxs.cpu().numpy().astype(np.int32)
    cls_cpu         = classifications.cpu().numpy()
    facing_cpu      = geo_props["facing"].cpu().numpy()
    half_cpu        = geo_props["half"].cpu().numpy()
    horiz_cpu       = geo_props["horiz_neighbors"].cpu().numpy()

    # ── 6. Resolve block state strings (CPU) ──
    block_states = _resolve_block_states(
        palette_cpu, cls_cpu, facing_cpu, half_cpu, horiz_cpu)

    # ── 7. Serialize (CPU) ──
    schem_bytes = _build_schem_bytes(coords_cpu, block_states)

    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(schem_bytes)

    return schem_bytes
