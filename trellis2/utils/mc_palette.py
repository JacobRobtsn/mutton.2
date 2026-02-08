"""
Minecraft block palette: maps block state strings to average top-face RGB colors.
GPU-accelerated nearest-neighbor color matching via brute-force L2 on CUDA.
"""
import torch
import numpy as np

# {block_state_string: (R, G, B)} — average top-face colors, 0-255
BLOCK_COLORS = {
    # ── Concrete (16) ──
    "minecraft:white_concrete":          (207, 213, 214),
    "minecraft:orange_concrete":         (224, 97, 1),
    "minecraft:magenta_concrete":        (169, 48, 159),
    "minecraft:light_blue_concrete":     (36, 137, 199),
    "minecraft:yellow_concrete":         (241, 175, 21),
    "minecraft:lime_concrete":           (94, 169, 25),
    "minecraft:pink_concrete":           (214, 101, 143),
    "minecraft:gray_concrete":           (55, 58, 62),
    "minecraft:light_gray_concrete":     (125, 125, 115),
    "minecraft:cyan_concrete":           (21, 119, 136),
    "minecraft:purple_concrete":         (100, 32, 156),
    "minecraft:blue_concrete":           (45, 47, 143),
    "minecraft:brown_concrete":          (96, 60, 32),
    "minecraft:green_concrete":          (73, 91, 36),
    "minecraft:red_concrete":            (142, 33, 33),
    "minecraft:black_concrete":          (8, 10, 15),

    # ── Wool (16) ──
    "minecraft:white_wool":              (234, 236, 236),
    "minecraft:orange_wool":             (241, 118, 20),
    "minecraft:magenta_wool":            (189, 68, 179),
    "minecraft:light_blue_wool":         (58, 175, 217),
    "minecraft:yellow_wool":             (249, 198, 40),
    "minecraft:lime_wool":               (112, 185, 26),
    "minecraft:pink_wool":               (238, 141, 172),
    "minecraft:gray_wool":               (63, 68, 72),
    "minecraft:light_gray_wool":         (142, 142, 135),
    "minecraft:cyan_wool":               (21, 138, 145),
    "minecraft:purple_wool":             (122, 42, 173),
    "minecraft:blue_wool":               (53, 57, 157),
    "minecraft:brown_wool":              (114, 72, 41),
    "minecraft:green_wool":              (85, 110, 28),
    "minecraft:red_wool":                (161, 39, 35),
    "minecraft:black_wool":              (20, 21, 26),

    # ── Terracotta / Stained Terracotta (17) ──
    "minecraft:terracotta":              (152, 94, 68),
    "minecraft:white_terracotta":        (210, 178, 161),
    "minecraft:orange_terracotta":       (162, 84, 38),
    "minecraft:magenta_terracotta":      (150, 88, 109),
    "minecraft:light_blue_terracotta":   (113, 109, 138),
    "minecraft:yellow_terracotta":       (186, 133, 35),
    "minecraft:lime_terracotta":         (104, 118, 53),
    "minecraft:pink_terracotta":         (162, 78, 79),
    "minecraft:gray_terracotta":         (58, 42, 36),
    "minecraft:light_gray_terracotta":   (135, 107, 98),
    "minecraft:cyan_terracotta":         (87, 91, 91),
    "minecraft:purple_terracotta":       (118, 70, 86),
    "minecraft:blue_terracotta":         (74, 60, 91),
    "minecraft:brown_terracotta":        (77, 51, 36),
    "minecraft:green_terracotta":        (76, 83, 42),
    "minecraft:red_terracotta":          (143, 61, 47),
    "minecraft:black_terracotta":        (37, 23, 16),

    # ── Glazed Terracotta (16) ──
    "minecraft:white_glazed_terracotta":        (188, 212, 202),
    "minecraft:orange_glazed_terracotta":       (154, 147, 91),
    "minecraft:magenta_glazed_terracotta":      (208, 100, 192),
    "minecraft:light_blue_glazed_terracotta":   (94, 164, 208),
    "minecraft:yellow_glazed_terracotta":       (234, 192, 88),
    "minecraft:lime_glazed_terracotta":         (162, 197, 55),
    "minecraft:pink_glazed_terracotta":         (235, 155, 181),
    "minecraft:gray_glazed_terracotta":         (83, 90, 93),
    "minecraft:light_gray_glazed_terracotta":   (144, 166, 167),
    "minecraft:cyan_glazed_terracotta":         (52, 118, 119),
    "minecraft:purple_glazed_terracotta":       (109, 48, 152),
    "minecraft:blue_glazed_terracotta":         (47, 65, 139),
    "minecraft:brown_glazed_terracotta":        (120, 106, 86),
    "minecraft:green_glazed_terracotta":        (163, 178, 55),
    "minecraft:red_glazed_terracotta":          (181, 60, 53),
    "minecraft:black_glazed_terracotta":        (67, 30, 32),

    # ── Stone variants ──
    "minecraft:stone":                   (126, 126, 126),
    "minecraft:cobblestone":             (127, 127, 127),
    "minecraft:stone_bricks":            (122, 121, 122),
    "minecraft:mossy_stone_bricks":      (115, 121, 105),
    "minecraft:cracked_stone_bricks":    (118, 117, 118),
    "minecraft:smooth_stone":            (161, 161, 161),
    "minecraft:granite":                 (149, 103, 86),
    "minecraft:polished_granite":        (154, 107, 89),
    "minecraft:diorite":                 (188, 188, 189),
    "minecraft:polished_diorite":        (192, 193, 194),
    "minecraft:andesite":                (136, 136, 136),
    "minecraft:polished_andesite":       (132, 135, 134),
    "minecraft:deepslate":               (80, 80, 82),
    "minecraft:cobbled_deepslate":       (77, 77, 80),
    "minecraft:polished_deepslate":      (72, 72, 73),
    "minecraft:deepslate_bricks":        (70, 70, 72),
    "minecraft:deepslate_tiles":         (54, 54, 55),
    "minecraft:tuff":                    (108, 109, 103),
    "minecraft:calcite":                 (224, 225, 221),
    "minecraft:dripstone_block":         (134, 107, 92),
    "minecraft:basalt":                  (73, 72, 78),
    "minecraft:polished_basalt":         (98, 98, 101),
    "minecraft:blackstone":              (42, 36, 41),
    "minecraft:polished_blackstone":     (53, 49, 56),
    "minecraft:end_stone":               (219, 222, 158),
    "minecraft:end_stone_bricks":        (218, 224, 162),
    "minecraft:purpur_block":            (170, 126, 170),

    # ── Sandstone / Red Sandstone ──
    "minecraft:sandstone":               (216, 203, 156),
    "minecraft:smooth_sandstone":        (223, 214, 170),
    "minecraft:red_sandstone":           (186, 99, 29),
    "minecraft:smooth_red_sandstone":    (181, 97, 31),

    # ── Nether blocks ──
    "minecraft:netherrack":              (97, 38, 38),
    "minecraft:nether_bricks":           (44, 21, 26),
    "minecraft:red_nether_bricks":       (69, 7, 9),
    "minecraft:soul_sand":               (81, 62, 50),
    "minecraft:soul_soil":               (75, 57, 46),
    "minecraft:magma_block":             (142, 63, 31),
    "minecraft:glowstone":               (171, 131, 73),
    "minecraft:shroomlight":             (240, 147, 57),
    "minecraft:crying_obsidian":         (32, 10, 60),
    "minecraft:obsidian":                (15, 11, 25),
    "minecraft:warped_nylium":           (43, 114, 101),
    "minecraft:crimson_nylium":          (130, 32, 32),

    # ── Wood planks ──
    "minecraft:oak_planks":              (162, 131, 78),
    "minecraft:spruce_planks":           (115, 85, 49),
    "minecraft:birch_planks":            (196, 179, 123),
    "minecraft:jungle_planks":           (160, 115, 81),
    "minecraft:acacia_planks":           (169, 92, 51),
    "minecraft:dark_oak_planks":         (67, 43, 20),
    "minecraft:mangrove_planks":         (117, 54, 48),
    "minecraft:cherry_planks":           (226, 178, 172),
    "minecraft:bamboo_planks":           (194, 175, 82),
    "minecraft:crimson_planks":          (101, 49, 71),
    "minecraft:warped_planks":           (43, 105, 99),

    # ── Metal / Ore blocks ──
    "minecraft:iron_block":              (220, 220, 220),
    "minecraft:gold_block":              (246, 208, 62),
    "minecraft:diamond_block":           (98, 237, 228),
    "minecraft:emerald_block":           (42, 176, 68),
    "minecraft:lapis_block":             (31, 67, 140),
    "minecraft:redstone_block":          (175, 24, 5),
    "minecraft:netherite_block":         (66, 61, 63),
    "minecraft:copper_block":            (192, 107, 79),
    "minecraft:exposed_copper":          (154, 121, 101),
    "minecraft:weathered_copper":        (109, 145, 107),
    "minecraft:oxidized_copper":         (82, 162, 132),
    "minecraft:amethyst_block":          (133, 97, 191),
    "minecraft:raw_iron_block":          (166, 136, 107),
    "minecraft:raw_gold_block":          (221, 169, 47),
    "minecraft:raw_copper_block":        (154, 105, 79),
    "minecraft:coal_block":              (16, 15, 15),

    # ── Prismarine ──
    "minecraft:prismarine":              (99, 156, 151),
    "minecraft:prismarine_bricks":       (99, 171, 158),
    "minecraft:dark_prismarine":         (51, 91, 75),

    # ── Misc ──
    "minecraft:quartz_block":            (236, 230, 223),
    "minecraft:smooth_quartz":           (236, 230, 223),
    "minecraft:bricks":                  (151, 98, 83),
    "minecraft:mud_bricks":              (137, 104, 79),
    "minecraft:packed_mud":              (142, 107, 82),
    "minecraft:clay":                    (160, 166, 179),
    "minecraft:packed_ice":              (141, 180, 224),
    "minecraft:blue_ice":                (116, 167, 253),
    "minecraft:snow_block":              (249, 254, 254),
    "minecraft:moss_block":              (89, 109, 45),
    "minecraft:hay_block":               (166, 139, 12),
    "minecraft:bone_block":              (229, 225, 207),
    "minecraft:honeycomb_block":         (229, 148, 29),
    "minecraft:slime_block":             (112, 192, 73),
    "minecraft:dried_kelp_block":        (50, 54, 30),
    "minecraft:sponge":                  (195, 192, 74),
    "minecraft:wet_sponge":              (171, 181, 70),
    "minecraft:melon":                   (111, 145, 30),
    "minecraft:pumpkin":                 (198, 119, 24),
    "minecraft:ochre_froglight":         (247, 224, 168),
    "minecraft:verdant_froglight":       (209, 240, 204),
    "minecraft:pearlescent_froglight":   (245, 222, 242),

    # ── Glass (transparent, but useful for builds) ──
    "minecraft:white_stained_glass":     (255, 255, 255),
    "minecraft:orange_stained_glass":    (216, 127, 51),
    "minecraft:magenta_stained_glass":   (178, 76, 216),
    "minecraft:light_blue_stained_glass":(102, 153, 216),
    "minecraft:yellow_stained_glass":    (229, 229, 51),
    "minecraft:lime_stained_glass":      (127, 204, 25),
    "minecraft:pink_stained_glass":      (242, 127, 165),
    "minecraft:cyan_stained_glass":      (76, 127, 153),
    "minecraft:purple_stained_glass":    (127, 63, 178),
    "minecraft:blue_stained_glass":      (51, 76, 178),
    "minecraft:brown_stained_glass":     (102, 76, 51),
    "minecraft:green_stained_glass":     (102, 127, 51),
    "minecraft:red_stained_glass":       (153, 51, 51),
    "minecraft:black_stained_glass":     (25, 25, 25),
}

# Precompute lists (order-stable)
BLOCK_NAMES = list(BLOCK_COLORS.keys())
_PALETTE_NP = np.array([BLOCK_COLORS[b] for b in BLOCK_NAMES], dtype=np.float32)

# Lazy-init GPU tensor (created on first use, cached per device)
_palette_cache = {}


def _get_palette_gpu(device):
    """Get palette tensor on the given CUDA device (cached)."""
    if device not in _palette_cache:
        _palette_cache[device] = torch.tensor(
            _PALETTE_NP, dtype=torch.float32, device=device
        )  # (K, 3)
    return _palette_cache[device]


@torch.no_grad()
def match_colors_gpu(colors: torch.Tensor) -> torch.Tensor:
    """
    GPU brute-force nearest-neighbor: for each input color find the closest
    palette entry by squared L2 distance.
    
    Args:
        colors: (N, 3) float tensor on CUDA, values 0-255
    
    Returns:
        (N,) int64 tensor of palette indices on same device
    """
    palette = _get_palette_gpu(colors.device)  # (K, 3)
    # Squared L2 via expansion: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    # This avoids materializing the full (N, K, 3) difference tensor
    a_sq = (colors * colors).sum(dim=1, keepdim=True)      # (N, 1)
    b_sq = (palette * palette).sum(dim=1, keepdim=True).T   # (1, K)
    dots = colors @ palette.T                                # (N, K)
    dists = a_sq + b_sq - 2.0 * dots                        # (N, K)
    return dists.argmin(dim=1)                               # (N,)
