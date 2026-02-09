"""
GPU-accelerated geometry analysis for Minecraft special block placement.

Detects surface edges and classifies voxels as slabs, stairs, fences, walls,
doors, or trapdoors based on neighbor occupancy patterns.
"""
import torch

# ── Classification constants ──
FULL_BLOCK  = 0
BOTTOM_SLAB = 1
TOP_SLAB    = 2
STAIR       = 3
FENCE       = 4
WALL        = 5
DOOR_LOWER  = 6
DOOR_UPPER  = 7
TRAPDOOR    = 8

# Neighbor indices (TRELLIS coord space → MC directions)
# TRELLIS col0=X(MC X), col1=Y(MC Z), col2=Z(MC Y)
_EAST  = 0   # +col0
_WEST  = 1   # -col0
_SOUTH = 2   # +col1 → +MC_Z
_NORTH = 3   # -col1 → -MC_Z
_UP    = 4   # +col2 → +MC_Y
_DOWN  = 5   # -col2 → -MC_Y

_HORIZ = [_EAST, _WEST, _SOUTH, _NORTH]

_FACING_STR = {_EAST: "east", _WEST: "west", _SOUTH: "south", _NORTH: "north"}
_OPPOSITE   = {_EAST: _WEST, _WEST: _EAST, _SOUTH: _NORTH, _NORTH: _SOUTH}


# ─────────────────────────────────────────────
# Occupancy grid
# ─────────────────────────────────────────────

def build_occupancy_grid(coords: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Sparse coords → dense bool (S,S,S) grid on same device."""
    grid = torch.zeros(grid_size, grid_size, grid_size,
                       dtype=torch.bool, device=coords.device)
    c = coords.long()
    grid[c[:, 0], c[:, 1], c[:, 2]] = True
    return grid


# ─────────────────────────────────────────────
# Neighbor masks
# ─────────────────────────────────────────────

def compute_neighbor_masks(grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    For each occupied voxel, compute 6-direction neighbor presence.

    Returns (N, 6) bool: [+X, -X, +Y, -Y, +Z, -Z].
    """
    S = grid.shape[0]
    c = coords.long()
    x, y, z = c[:, 0], c[:, 1], c[:, 2]

    def _lookup(dx, dy, dz):
        nx = (x + dx).clamp(0, S - 1)
        ny = (y + dy).clamp(0, S - 1)
        nz = (z + dz).clamp(0, S - 1)
        hit = grid[nx, ny, nz]
        # Out-of-bounds → False
        if dx > 0:  hit = hit & (x < S - 1)
        elif dx < 0: hit = hit & (x > 0)
        if dy > 0:  hit = hit & (y < S - 1)
        elif dy < 0: hit = hit & (y > 0)
        if dz > 0:  hit = hit & (z < S - 1)
        elif dz < 0: hit = hit & (z > 0)
        return hit

    return torch.stack([
        _lookup( 1, 0, 0),   # +X  east
        _lookup(-1, 0, 0),   # -X  west
        _lookup( 0, 1, 0),   # +Y  south
        _lookup( 0,-1, 0),   # -Y  north
        _lookup( 0, 0, 1),   # +Z  up
        _lookup( 0, 0,-1),   # -Z  down
    ], dim=1)  # (N, 6)


# ─────────────────────────────────────────────
# Classification heuristics
# ─────────────────────────────────────────────

@torch.no_grad()
def classify_voxels(neighbors: torch.Tensor, detect_doors: bool = False):
    """
    Classify each voxel from its neighbor mask.

    Returns:
        cls           (N,) int32  — classification constant
        facing        (N,) int32  — horizontal direction index
        half          (N,) int32  — 0=bottom, 1=top
        horiz_neighbors (N, 4) bool — [E, W, S, N]
    """
    N = neighbors.shape[0]
    device = neighbors.device

    cls    = torch.full((N,), FULL_BLOCK, dtype=torch.int32, device=device)
    facing = torch.full((N,), _NORTH,     dtype=torch.int32, device=device)
    half   = torch.zeros(N, dtype=torch.int32, device=device)

    has_up   = neighbors[:, _UP]
    has_down = neighbors[:, _DOWN]

    horiz = neighbors[:, [_EAST, _WEST, _SOUTH, _NORTH]]  # (N, 4)
    n_horiz     = horiz.sum(dim=1)      # solid horizontal count
    n_horiz_air = 4 - n_horiz           # open horizontal count

    n_total    = neighbors.sum(dim=1)
    is_surface = n_total < 6

    # ── Bottom slab: air above, solid below, 1-2 open horizontal sides ──
    bottom_slab = is_surface & (~has_up) & has_down & (n_horiz_air >= 1) & (n_horiz_air <= 2)
    cls[bottom_slab] = BOTTOM_SLAB

    # ── Top slab: solid above, air below, 1-2 open horizontal sides ──
    top_slab = is_surface & has_up & (~has_down) & (n_horiz_air >= 1) & (n_horiz_air <= 2)
    cls[top_slab] = TOP_SLAB
    half[top_slab] = 1

    # ── Stair: air above, solid below, exactly 1 open horizontal (overrides slab) ──
    stair = is_surface & (~has_up) & has_down & (n_horiz_air == 1)
    open_horiz = ~horiz                 # True where air
    horiz_dirs = torch.tensor([_EAST, _WEST, _SOUTH, _NORTH], device=device)
    open_dir_idx   = open_horiz.float().argmax(dim=1)   # index of the open side
    stair_facing   = horiz_dirs[open_dir_idx]
    cls[stair]     = STAIR
    facing[stair]  = stair_facing[stair]

    # ── Fence / Wall: ≤2 horizontal neighbors in a linear pattern ──
    opp_ew = horiz[:, 0] & horiz[:, 1]
    opp_ns = horiz[:, 2] & horiz[:, 3]
    is_linear = (n_horiz <= 1) | ((n_horiz == 2) & (opp_ew | opp_ns))
    fence_wall = is_surface & is_linear & (n_horiz >= 1) & (n_horiz <= 2) & (cls == FULL_BLOCK)
    cls[fence_wall] = FENCE             # resolved to fence or wall by material

    # ── Door / Trapdoor (opt-in) ──
    if detect_doors:
        door = is_surface & (n_horiz <= 1) & has_up & (~has_down) & (cls == FULL_BLOCK)
        cls[door] = DOOR_LOWER

        trapdoor = is_surface & (n_total <= 2) & (n_horiz <= 1) & (cls == FULL_BLOCK)
        cls[trapdoor] = TRAPDOOR

    return cls, facing, half, horiz


# ─────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────

@torch.no_grad()
def classify_and_resolve(coords: torch.Tensor, grid_size: int,
                         detect_doors: bool = False):
    """
    Full GPU geometry pipeline.

    Returns:
        cls   (N,) int32
        props dict  — 'facing', 'half', 'horiz_neighbors'
    """
    grid      = build_occupancy_grid(coords, grid_size)
    neighbors = compute_neighbor_masks(grid, coords)
    cls, facing, half, horiz = classify_voxels(neighbors, detect_doors=detect_doors)
    return cls, {"facing": facing, "half": half, "horiz_neighbors": horiz}
