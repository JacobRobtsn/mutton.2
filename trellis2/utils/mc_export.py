"""
Convert TRELLIS.2 MeshWithVoxel output to Minecraft .schem (Sponge Schematic v2).
"""
import io
import gzip
import struct
from typing import Optional, Dict
import numpy as np
import torch

from .mc_palette import match_colors_batch


# ──────────────────────────────────────────────
# Minimal NBT writer (avoids external dependency)
# Sponge Schematic v2 spec uses Named Binary Tag format
# ──────────────────────────────────────────────

TAG_END = 0
TAG_BYTE = 1
TAG_SHORT = 2
TAG_INT = 3
TAG_LONG = 4
TAG_FLOAT = 5
TAG_DOUBLE = 6
TAG_BYTE_ARRAY = 7
TAG_STRING = 8
TAG_LIST = 9
TAG_COMPOUND = 10
TAG_INT_ARRAY = 11
TAG_LONG_ARRAY = 12


def _write_tag_header(buf, tag_type, name):
    buf.write(struct.pack(">b", tag_type))
    encoded = name.encode("utf-8")
    buf.write(struct.pack(">H", len(encoded)))
    buf.write(encoded)


def _write_byte(buf, name, val):
    _write_tag_header(buf, TAG_BYTE, name)
    buf.write(struct.pack(">b", val))


def _write_short(buf, name, val):
    _write_tag_header(buf, TAG_SHORT, name)
    buf.write(struct.pack(">h", val))


def _write_int(buf, name, val):
    _write_tag_header(buf, TAG_INT, name)
    buf.write(struct.pack(">i", val))


def _write_string(buf, name, val):
    _write_tag_header(buf, TAG_STRING, name)
    encoded = val.encode("utf-8")
    buf.write(struct.pack(">H", len(encoded)))
    buf.write(encoded)


def _write_byte_array(buf, name, data: bytes):
    _write_tag_header(buf, TAG_BYTE_ARRAY, name)
    buf.write(struct.pack(">i", len(data)))
    buf.write(data)


def _write_int_array(buf, name, vals):
    _write_tag_header(buf, TAG_INT_ARRAY, name)
    buf.write(struct.pack(">i", len(vals)))
    for v in vals:
        buf.write(struct.pack(">i", v))


def _write_compound_start(buf, name):
    _write_tag_header(buf, TAG_COMPOUND, name)


def _write_compound_end(buf):
    buf.write(struct.pack(">b", TAG_END))


def _encode_varint(value):
    """Encode an integer as a varint (used by Sponge Schematic BlockData)."""
    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value != 0:
            byte |= 0x80
        result.append(byte)
        if value == 0:
            break
    return bytes(result)


# ──────────────────────────────────────────────
# Core export functions
# ──────────────────────────────────────────────

def _downsample_voxels(coords, colors, native_res, target_res):
    """
    Downsample sparse voxels by binning into a coarser grid and averaging colors.
    
    Args:
        coords: (N, 3) int numpy array, voxel positions in [0, native_res)
        colors: (N, 3) float numpy array, RGB in [0, 1]
        native_res: int, original grid resolution
        target_res: int, desired grid resolution (must be <= native_res)
    
    Returns:
        new_coords: (M, 3) int array
        new_colors: (M, 3) float array
    """
    if target_res >= native_res:
        return coords, colors

    ratio = native_res / target_res
    binned = (coords / ratio).astype(np.int32)
    binned = np.clip(binned, 0, target_res - 1)

    # Unique bins and average colors
    # Encode 3D coords to 1D keys
    keys = binned[:, 0] * target_res * target_res + binned[:, 1] * target_res + binned[:, 2]
    unique_keys, inverse = np.unique(keys, return_inverse=True)

    new_colors = np.zeros((len(unique_keys), 3), dtype=np.float64)
    counts = np.zeros(len(unique_keys), dtype=np.float64)
    np.add.at(new_colors, inverse, colors)
    np.add.at(counts, inverse, 1)
    new_colors /= counts[:, None]

    # Decode keys back to coords
    new_coords = np.stack([
        unique_keys // (target_res * target_res),
        (unique_keys // target_res) % target_res,
        unique_keys % target_res,
    ], axis=1).astype(np.int32)

    return new_coords, new_colors.astype(np.float32)


def voxels_to_schem_bytes(
    coords: np.ndarray,
    block_ids: list,
    y_up: bool = True,
) -> bytes:
    """
    Build a Sponge Schematic v2 .schem file from voxel coordinates and block IDs.
    
    Args:
        coords: (N, 3) int array of occupied voxel positions (x, y, z)
        block_ids: list of N block state strings (e.g. "minecraft:stone")
        y_up: if True, treat axis 1 as Y (height). TRELLIS.2 uses Y-up.
    
    Returns:
        gzipped bytes of the .schem file
    """
    coords = coords.copy()
    # Shift to zero-origin
    min_c = coords.min(axis=0)
    coords -= min_c

    max_c = coords.max(axis=0)
    # TRELLIS.2: axis order is (X, Y, Z) with Y up
    # .schem: Width=X, Height=Y, Length=Z
    width = int(max_c[0]) + 1
    height = int(max_c[1]) + 1
    length = int(max_c[2]) + 1

    # Build palette
    unique_blocks = list(set(block_ids))
    # Air is always index 0 if we need it
    if "minecraft:air" not in unique_blocks:
        unique_blocks.insert(0, "minecraft:air")
    palette_map = {b: i for i, b in enumerate(unique_blocks)}

    # Build 3D block data array (default to air = 0)
    total = width * height * length
    block_data_flat = np.zeros(total, dtype=np.int32)

    for i in range(len(coords)):
        x, y, z = int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2])
        # .schem index order: (y * length + z) * width + x
        idx = (y * length + z) * width + x
        block_data_flat[idx] = palette_map[block_ids[i]]

    # Varint encode block data
    varint_buf = bytearray()
    for val in block_data_flat:
        varint_buf.extend(_encode_varint(val))

    # Write NBT
    buf = io.BytesIO()
    _write_compound_start(buf, "Schematic")
    _write_int(buf, "Version", 2)
    _write_int(buf, "DataVersion", 3578)  # 1.20.4
    _write_short(buf, "Width", width)
    _write_short(buf, "Height", height)
    _write_short(buf, "Length", length)

    # Palette compound
    _write_compound_start(buf, "Palette")
    for block_name, idx in palette_map.items():
        _write_int(buf, block_name, idx)
    _write_compound_end(buf)

    _write_int(buf, "PaletteMax", len(palette_map))
    _write_byte_array(buf, "BlockData", bytes(varint_buf))

    # Metadata (optional but some tools expect it)
    _write_compound_start(buf, "Metadata")
    _write_string(buf, "Generator", "TRELLIS2-Minecraft")
    _write_compound_end(buf)

    # Offset
    _write_int_array(buf, "Offset", [0, 0, 0])

    _write_compound_end(buf)  # end Schematic

    # Gzip compress
    raw = buf.getvalue()
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb") as gz:
        gz.write(raw)
    return out.getvalue()


def to_schem(
    mesh_with_voxel,
    target_resolution: Optional[int] = None,
    output_path: Optional[str] = None,
) -> bytes:
    """
    Convert a TRELLIS.2 MeshWithVoxel to a Minecraft .schem file.
    
    Args:
        mesh_with_voxel: MeshWithVoxel from pipeline output
        target_resolution: optional int, downsample voxels to this grid size.
            If None, uses native resolution.
        output_path: if provided, write .schem file to this path
    
    Returns:
        bytes of the .schem file (gzipped NBT)
    """
    m = mesh_with_voxel

    # Extract coords and base_color from sparse voxel attrs
    coords_t = m.coords  # (N, 3) int tensor
    attrs_t = m.attrs     # (N, C) float tensor
    color_slice = m.layout.get("base_color", slice(0, 3))
    colors_t = attrs_t[:, color_slice]  # (N, 3) in [0, 1]

    coords_np = coords_t.cpu().numpy().astype(np.int32)
    colors_np = colors_t.cpu().numpy().astype(np.float32)

    # Compute native resolution from coords
    native_res = int(coords_np.max()) + 1

    # Optional downsample
    if target_resolution is not None and target_resolution < native_res:
        coords_np, colors_np = _downsample_voxels(
            coords_np, colors_np, native_res, target_resolution
        )

    # Map colors (0-1) to block IDs via palette
    colors_255 = np.clip(colors_np * 255.0, 0, 255)
    block_ids = match_colors_batch(colors_255)

    # Build .schem
    schem_bytes = voxels_to_schem_bytes(coords_np, block_ids)

    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(schem_bytes)

    return schem_bytes
