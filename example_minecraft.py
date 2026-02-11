"""
TRELLIS.2 → Minecraft .schem export example.
Generates a 3D asset from an image and saves it as a Sponge Schematic v2 file
that can be loaded in Minecraft via WorldEdit / FAWE / Litematica.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils.mc_export import to_schem

# ── Config ──
IMAGE_PATH = "assets/example_image/T.png"
OUTPUT_PATH = "sample.schem"
TARGET_RESOLUTION = 64       # Minecraft build size per axis (None = native)
PIPELINE_TYPE = "512"        # "512" is fastest and sufficient for block builds

# ── Run ──
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

image = Image.open(IMAGE_PATH)
mesh = pipeline.run(image, pipeline_type=PIPELINE_TYPE)[0]

schem_bytes = to_schem(
    mesh,
    target_resolution=TARGET_RESOLUTION,
    output_path=OUTPUT_PATH,
    denoise_iterations=5,       # 3-5 eliminates color noise; 0 = off
)

print(f"Saved {OUTPUT_PATH} ({len(schem_bytes)} bytes)")
print(f"Load in Minecraft: //schem load {os.path.splitext(os.path.basename(OUTPUT_PATH))[0]}")
