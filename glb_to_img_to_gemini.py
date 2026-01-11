# #!/usr/bin/env python3
# """
# glb_render_and_send_to_gemini.py

# 1) Render a .glb asset pack to PNGs using Blender (headless).
# 2) Send each rendered PNG to Gemini with a hardcoded prompt.
# 3) Save responses to a JSONL file (one line per image).

# Requirements:
#   - Blender installed
#   - pip install -U google-genai
#   - export GEMINI_API_KEY="..."

# Usage:
#   python glb_render_and_send_to_gemini.py /path/to/assetpack.glb --out renders

# Example (minimum 2 angles):
#   python glb_render_and_send_to_gemini.py your.glb --views "45,20;225,20" --out renders

# More angles (better occlusion coverage):
#   python glb_render_and_send_to_gemini.py your.glb --views "45,20;135,20;225,20;315,20;45,60;225,60" --out renders
# """

# import argparse
# import glob
# import json
# import os
# import subprocess
# import sys
# import tempfile
# import time
# from typing import List, Optional

# from google import genai
# from google.genai import types


# # ============================================================
# # 1) HARD-CODE YOUR GEMINI PROMPT HERE
# # ============================================================
# GEMINI_PROMPT = "Given these images of a GLB 3D asset pack, carefully identify the number of distinct objects present in the pack. An object is defined as a separate, individual item that can be distinguished from others based on its geometry and structure. Consider factors such as connectivity, spatial separation, and unique features when determining object boundaries. Provide only the total count of distinct objects without any additional explanation."
# # PASTE YOUR PROMPT HERE.

# # Example:
# # You will be shown a rendered view of a 3D asset pack containing multiple objects.
# # Estimate how many distinct, separate objects are present in the image.
# # Return ONLY JSON with keys: object_count (int), confidence (0-1), notes (string).
# # """


# # ============================================================
# # Blender script embedded (renders multi-view images)
# # ============================================================
# BLENDER_RENDER_SCRIPT = r"""
# import bpy
# import sys
# import os
# import math
# import mathutils

# argv = sys.argv
# argv = argv[argv.index("--") + 1:] if "--" in argv else []

# def parse_args(argv):
#     out = {}
#     it = iter(argv)
#     for tok in it:
#         if tok.startswith("--"):
#             out[tok[2:]] = next(it)
#     return out

# args = parse_args(argv)
# glb_path = args["glb"]
# out_dir  = args["out"]
# width    = int(args.get("w", "1280"))
# height   = int(args.get("h", "1280"))
# views_s  = args.get("views", "45,20;225,20;45,60;225,60")
# engine   = args.get("engine", "EEVEE").upper()
# transparent = args.get("transparent", "1") == "1"
# bg_strength = float(args.get("bg_strength", "0.6"))
# samples = int(args.get("samples", "64"))

# os.makedirs(out_dir, exist_ok=True)

# # Reset to an empty scene
# bpy.ops.wm.read_factory_settings(use_empty=True)

# # Import GLB
# bpy.ops.import_scene.gltf(filepath=glb_path)

# mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
# if not mesh_objs:
#     raise RuntimeError("No mesh objects found after importing GLB.")

# # Compute world-space bounds across all meshes
# min_v = mathutils.Vector(( 1e30,  1e30,  1e30))
# max_v = mathutils.Vector((-1e30, -1e30, -1e30))

# for obj in mesh_objs:
#     for corner in obj.bound_box:
#         v = obj.matrix_world @ mathutils.Vector(corner)
#         min_v.x = min(min_v.x, v.x); min_v.y = min(min_v.y, v.y); min_v.z = min(min_v.z, v.z)
#         max_v.x = max(max_v.x, v.x); max_v.y = max(max_v.y, v.y); max_v.z = max(max_v.z, v.z)

# center = (min_v + max_v) * 0.5
# diag = (max_v - min_v).length
# radius = max(1e-6, diag * 0.5)

# # Create camera
# cam_data = bpy.data.cameras.new("Camera")
# cam = bpy.data.objects.new("Camera", cam_data)
# bpy.context.scene.collection.objects.link(cam)
# bpy.context.scene.camera = cam

# # Set camera vertical FOV
# cam.data.lens_unit = 'FOV'
# cam.data.angle = math.radians(45.0)

# # Target empty to track
# target = bpy.data.objects.new("Target", None)
# target.empty_display_type = 'PLAIN_AXES'
# target.empty_display_size = max(radius * 0.05, 0.01)
# target.location = center
# bpy.context.scene.collection.objects.link(target)

# track = cam.constraints.new(type='TRACK_TO')
# track.target = target
# track.track_axis = 'TRACK_NEGATIVE_Z'
# track.up_axis = 'UP_Y'

# # World light
# world = bpy.data.worlds.new("World")
# world.use_nodes = True
# bpy.context.scene.world = world
# bg = world.node_tree.nodes.get("Background")
# bg.inputs[1].default_value = bg_strength

# # Add sun lights
# def add_sun(name, az_deg, el_deg, energy):
#     light_data = bpy.data.lights.new(name=name, type='SUN')
#     light_data.energy = energy
#     light_obj = bpy.data.objects.new(name=name, object_data=light_data)
#     bpy.context.scene.collection.objects.link(light_obj)

#     az = math.radians(az_deg)
#     el = math.radians(el_deg)
#     dist = radius * 3.0
#     x = center.x + dist * math.cos(el) * math.cos(az)
#     y = center.y + dist * math.sin(el)
#     z = center.z + dist * math.cos(el) * math.sin(az)
#     light_obj.location = (x, y, z)

#     direction = mathutils.Vector((center.x - x, center.y - y, center.z - z))
#     rot_quat = direction.to_track_quat('-Z', 'Y')
#     light_obj.rotation_euler = rot_quat.to_euler()

# add_sun("Key", 45, 55, 3.0)
# add_sun("Fill", 225, 35, 2.0)

# # Render settings
# scene = bpy.context.scene
# scene.render.resolution_x = width
# scene.render.resolution_y = height
# scene.render.resolution_percentage = 100
# scene.render.image_settings.file_format = 'PNG'
# scene.render.film_transparent = transparent

# if engine == "CYCLES":
#     scene.render.engine = "CYCLES"
#     scene.cycles.samples = samples
# else:
#     scene.render.engine = "BLENDER_EEVEE"
#     scene.eevee.taa_render_samples = samples

# # Place camera so bounding sphere fits in vertical FOV
# def set_camera_view(az_deg, el_deg):
#     az = math.radians(az_deg)
#     el = math.radians(el_deg)
#     fov = cam.data.angle
#     dist = radius / math.sin(fov * 0.5)
#     dist *= 1.25  # margin

#     x = center.x + dist * math.cos(el) * math.cos(az)
#     y = center.y + dist * math.sin(el)
#     z = center.z + dist * math.cos(el) * math.sin(az)
#     cam.location = (x, y, z)

# views = []
# for pair in views_s.split(";"):
#     pair = pair.strip()
#     if not pair:
#         continue
#     az_s, el_s = pair.split(",")
#     views.append((float(az_s), float(el_s)))

# for i, (az, el) in enumerate(views):
#     set_camera_view(az, el)
#     out_path = os.path.join(out_dir, f"view_{i:02d}_az{int(az)}_el{int(el)}.png")
#     scene.render.filepath = out_path
#     bpy.ops.render.render(write_still=True)

# print(f"Done. Rendered {len(views)} images to {out_dir}")
# """


# # ============================================================
# # Rendering (Blender)
# # ============================================================
# def run_blender_render(
#     blender_path: str,
#     glb_path: str,
#     out_dir: str,
#     width: int,
#     height: int,
#     views: str,
#     engine: str,
#     transparent: bool,
#     bg_strength: float,
#     samples: int,
# ) -> None:
#     glb_path = os.path.abspath(glb_path)
#     out_dir = os.path.abspath(out_dir)
#     os.makedirs(out_dir, exist_ok=True)

#     with tempfile.TemporaryDirectory() as td:
#         script_path = os.path.join(td, "blender_render_script.py")
#         with open(script_path, "w", encoding="utf-8") as f:
#             f.write(BLENDER_RENDER_SCRIPT)

#         cmd = [
#             blender_path,
#             "-b",
#             "-P", script_path,
#             "--",
#             "--glb", glb_path,
#             "--out", out_dir,
#             "--w", str(width),
#             "--h", str(height),
#             "--views", views,
#             "--engine", engine,
#             "--transparent", "1" if transparent else "0",
#             "--bg_strength", str(bg_strength),
#             "--samples", str(samples),
#         ]

#         print("Running Blender:\n  " + " ".join(cmd))
#         subprocess.check_call(cmd)


# def list_rendered_pngs(out_dir: str) -> List[str]:
#     paths = sorted(glob.glob(os.path.join(out_dir, "view_*.png")))
#     return paths


# # ============================================================
# # Gemini sending
# # ============================================================
# def load_bytes(path: str) -> bytes:
#     with open(path, "rb") as f:
#         return f.read()


# def gemini_client(api_key: Optional[str] = None) -> genai.Client:
#     # If GEMINI_API_KEY is set, you can just do genai.Client().
#     if api_key:
#         return genai.Client(api_key=api_key)
#     return genai.Client()


# def send_image_to_gemini(
#     client: genai.Client,
#     model: str,
#     prompt: str,
#     image_path: str,
#     mime_type: str = "image/png",
# ):
#     img_bytes = load_bytes(image_path)
#     img_part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)

#     # Multimodal: contents can be a list of [text, image_part]
#     resp = client.models.generate_content(
#         model=model,
#         contents=[prompt, img_part],
#     )
#     return resp


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("glb", help="Path to .glb asset pack")
#     ap.add_argument("--out", default="glb_renders", help="Output directory")
#     ap.add_argument("--w", type=int, default=1280)
#     ap.add_argument("--h", type=int, default=1280)
#     ap.add_argument(
#         "--views",
#         default="45,20;225,20;45,60;225,60",
#         help='Semicolon-separated "az,el" degree pairs. Example: "45,20;225,20"',
#     )
#     ap.add_argument("--engine", default="EEVEE", choices=["EEVEE", "CYCLES"])
#     ap.add_argument("--samples", type=int, default=64)
#     ap.add_argument("--bg-strength", type=float, default=0.6)

#     ap.add_argument("--transparent", dest="transparent", action="store_true")
#     ap.add_argument("--no-transparent", dest="transparent", action="store_false")
#     ap.set_defaults(transparent=True)

#     ap.add_argument(
#         "--blender",
#         default="/Applications/Blender.app/Contents/MacOS/Blender",
#         help="Path to Blender executable",
#     )

#     # Gemini options
#     ap.add_argument("--model", default="gemini-3-flash-preview" \
#     "", help="Gemini model name")
#     ap.add_argument(
#         "--api-key",
#         default=None,
#         help="Optional API key override (otherwise uses GEMINI_API_KEY env var)",
#     )
#     ap.add_argument(
#         "--results",
#         default=None,
#         help="Output JSONL file for Gemini results (default: <out>/gemini_results.jsonl)",
#     )
#     ap.add_argument(
#         "--sleep",
#         type=float,
#         default=0.0,
#         help="Seconds to sleep between requests (rate-limit friendliness)",
#     )
#     ap.add_argument(
#         "--skip-existing",
#         action="store_true",
#         help="If set, skip sending images already present in the results JSONL",
#     )

#     args = ap.parse_args()

#     if not os.path.exists(args.blender):
#         print(f"ERROR: Blender not found at: {args.blender}", file=sys.stderr)
#         sys.exit(2)

#     # 1) Render
#     run_blender_render(
#         blender_path=args.blender,
#         glb_path=args.glb,
#         out_dir=args.out,
#         width=args.w,
#         height=args.h,
#         views=args.views,
#         engine=args.engine,
#         transparent=args.transparent,
#         bg_strength=args.bg_strength,
#         samples=args.samples,
#     )

#     pngs = list_rendered_pngs(args.out)
#     if not pngs:
#         print("ERROR: No rendered PNGs found.", file=sys.stderr)
#         sys.exit(3)

#     # 2) Gemini client
#     client = gemini_client(api_key=args.api_key)

#     # 3) Results output
#     results_path = args.results or os.path.join(args.out, "gemini_results.jsonl")

#     seen = set()
#     if args.skip_existing and os.path.exists(results_path):
#         try:
#             with open(results_path, "r", encoding="utf-8") as f:
#                 for line in f:
#                     line = line.strip()
#                     if not line:
#                         continue
#                     obj = json.loads(line)
#                     if "image_path" in obj:
#                         seen.add(obj["image_path"])
#         except Exception:
#             # If file is malformed, just don't skip.
#             seen = set()

#     # 4) Send each image
#     with open(results_path, "a", encoding="utf-8") as out_f:
#         for img_path in pngs:
#             abs_img = os.path.abspath(img_path)
#             if args.skip_existing and abs_img in seen:
#                 print(f"Skipping (already in results): {img_path}")
#                 continue

#             print(f"Sending to Gemini: {img_path}")
#             try:
#                 resp = send_image_to_gemini(
#                     client=client,
#                     model=args.model,
#                     prompt=GEMINI_PROMPT,
#                     image_path=img_path,
#                     mime_type="image/png",
#                 )
#                 # New (JSON-safe):
#                 raw = None
#                 if hasattr(resp, "model_dump"):
#                     raw = resp.model_dump(mode="json")  # <-- key change
#                 elif hasattr(resp, "to_dict"):
#                     raw = resp.to_dict()

#                 record = {
#                     "image_path": abs_img,
#                     "model": args.model,
#                     "prompt": GEMINI_PROMPT,
#                     "response_text": getattr(resp, "text", None),
#                     "raw": raw,
#                     "ts": time.time(),
#             }
#                 # record = {
#                 #     "image_path": abs_img,
#                 #     "model": args.model,
#                 #     "prompt": GEMINI_PROMPT,
#                 #     "response_text": getattr(resp, "text", None),
#                 #     "raw": resp.model_dump() if hasattr(resp, "model_dump") else None,
#                 #     "ts": time.time(),
#                 # }
#                 out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
#                 out_f.flush()

#             except Exception as e:
#                 err_record = {
#                     "image_path": abs_img,
#                     "model": args.model,
#                     "prompt": GEMINI_PROMPT,
#                     "error": repr(e),
#                     "ts": time.time(),
#                 }
#                 out_f.write(json.dumps(err_record, ensure_ascii=False) + "\n")
#                 out_f.flush()
#                 print(f"ERROR sending {img_path}: {e}", file=sys.stderr)

#             if args.sleep > 0:
#                 time.sleep(args.sleep)

#     print(f"\nDone. Results written to: {results_path}")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
glb_render_and_send_to_gemini_one_prompt.py

1) Render a .glb asset pack to multiple PNG images using Blender (headless).
2) Send ALL rendered PNGs to Gemini in ONE request with ONE prompt.
3) Labels each image in the prompt (recommended).
4) Saves one combined response record to JSONL.

Requirements:
  - Blender installed
  - pip install -U google-genai
  - export GEMINI_API_KEY="..."

Usage:
  python glb_render_and_send_to_gemini_one_prompt.py /path/to/assetpack.glb --out renders

Example (minimum 2 angles):
  python glb_render_and_send_to_gemini_one_prompt.py your.glb --views "45,20;225,20" --out renders

More angles:
  python glb_render_and_send_to_gemini_one_prompt.py your.glb --views "45,20;135,20;225,20;315,20;45,60;225,60" --out renders
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import List, Optional

from google import genai
from google.genai import types

GEMINI_PROMPT = "Given these images of a GLB 3D asset pack, carefully identify the number of distinct objects present in the pack. An object is defined as a separate, individual item that can be distinguished from others based on its geometry and structure. Consider factors such as connectivity, spatial separation, and unique features when determining object boundaries. Provide only the total count of distinct objects without any additional explanation."


# ============================================================
# Blender script embedded (renders multi-view images)
# ============================================================
BLENDER_RENDER_SCRIPT = r"""
import bpy
import sys
import os
import math
import mathutils

argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

def parse_args(argv):
    out = {}
    it = iter(argv)
    for tok in it:
        if tok.startswith("--"):
            out[tok[2:]] = next(it)
    return out

args = parse_args(argv)
glb_path = args["glb"]
out_dir  = args["out"]
width    = int(args.get("w", "896"))
height   = int(args.get("h", "896"))
views_s  = args.get("views", "45,20;225,20;45,60;225,60")
engine   = args.get("engine", "EEVEE").upper()
transparent = args.get("transparent", "1") == "1"
bg_strength = float(args.get("bg_strength", "0.6"))
samples = int(args.get("samples", "64"))

os.makedirs(out_dir, exist_ok=True)

# Reset to an empty scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import GLB
bpy.ops.import_scene.gltf(filepath=glb_path)

mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not mesh_objs:
    raise RuntimeError("No mesh objects found after importing GLB.")

# Compute world-space bounds across all meshes
min_v = mathutils.Vector(( 1e30,  1e30,  1e30))
max_v = mathutils.Vector((-1e30, -1e30, -1e30))

for obj in mesh_objs:
    for corner in obj.bound_box:
        v = obj.matrix_world @ mathutils.Vector(corner)
        min_v.x = min(min_v.x, v.x); min_v.y = min(min_v.y, v.y); min_v.z = min(min_v.z, v.z)
        max_v.x = max(max_v.x, v.x); max_v.y = max(max_v.y, v.y); max_v.z = max(max_v.z, v.z)

center = (min_v + max_v) * 0.5
diag = (max_v - min_v).length
radius = max(1e-6, diag * 0.5)

# Create camera
cam_data = bpy.data.cameras.new("Camera")
cam = bpy.data.objects.new("Camera", cam_data)
bpy.context.scene.collection.objects.link(cam)
bpy.context.scene.camera = cam

# Set camera vertical FOV
cam.data.lens_unit = 'FOV'
cam.data.angle = math.radians(45.0)

# Target empty to track
target = bpy.data.objects.new("Target", None)
target.empty_display_type = 'PLAIN_AXES'
target.empty_display_size = max(radius * 0.05, 0.01)
target.location = center
bpy.context.scene.collection.objects.link(target)

track = cam.constraints.new(type='TRACK_TO')
track.target = target
track.track_axis = 'TRACK_NEGATIVE_Z'
track.up_axis = 'UP_Y'

# World light
world = bpy.data.worlds.new("World")
world.use_nodes = True
bpy.context.scene.world = world
bg = world.node_tree.nodes.get("Background")
bg.inputs[1].default_value = bg_strength

# Add sun lights
def add_sun(name, az_deg, el_deg, energy):
    light_data = bpy.data.lights.new(name=name, type='SUN')
    light_data.energy = energy
    light_obj = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.scene.collection.objects.link(light_obj)

    az = math.radians(az_deg)
    el = math.radians(el_deg)
    dist = radius * 3.0
    x = center.x + dist * math.cos(el) * math.cos(az)
    y = center.y + dist * math.sin(el)
    z = center.z + dist * math.cos(el) * math.sin(az)
    light_obj.location = (x, y, z)

    direction = mathutils.Vector((center.x - x, center.y - y, center.z - z))
    rot_quat = direction.to_track_quat('-Z', 'Y')
    light_obj.rotation_euler = rot_quat.to_euler()

add_sun("Key", 45, 55, 3.0)
add_sun("Fill", 225, 35, 2.0)

# Render settings
scene = bpy.context.scene
scene.render.resolution_x = width
scene.render.resolution_y = height
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.film_transparent = transparent

if engine == "CYCLES":
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
else:
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = samples

# Place camera so bounding sphere fits in vertical FOV
def set_camera_view(az_deg, el_deg):
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    fov = cam.data.angle
    dist = radius / math.sin(fov * 0.5)
    dist *= 1.25  # margin

    x = center.x + dist * math.cos(el) * math.cos(az)
    y = center.y + dist * math.sin(el)
    z = center.z + dist * math.cos(el) * math.sin(az)
    cam.location = (x, y, z)

views = []
for pair in views_s.split(";"):
    pair = pair.strip()
    if not pair:
        continue
    az_s, el_s = pair.split(",")
    views.append((float(az_s), float(el_s)))

for i, (az, el) in enumerate(views):
    set_camera_view(az, el)
    out_path = os.path.join(out_dir, f"view_{i:02d}_az{int(az)}_el{int(el)}.png")
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)

print(f"Done. Rendered {len(views)} images to {out_dir}")
"""


# ============================================================
# Rendering (Blender)
# ============================================================
def run_blender_render(
    blender_path: str,
    glb_path: str,
    out_dir: str,
    width: int,
    height: int,
    views: str,
    engine: str,
    transparent: bool,
    bg_strength: float,
    samples: int,
) -> None:
    glb_path = os.path.abspath(glb_path)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        script_path = os.path.join(td, "blender_render_script.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(BLENDER_RENDER_SCRIPT)

        cmd = [
            blender_path,
            "-b",
            "-P", script_path,
            "--",
            "--glb", glb_path,
            "--out", out_dir,
            "--w", str(width),
            "--h", str(height),
            "--views", views,
            "--engine", engine,
            "--transparent", "1" if transparent else "0",
            "--bg_strength", str(bg_strength),
            "--samples", str(samples),
        ]

        print("Running Blender:\n  " + " ".join(cmd))
        subprocess.check_call(cmd)


def list_rendered_pngs(out_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(out_dir, "view_*.png")))


# ============================================================
# Gemini (one prompt, many images)
# ============================================================
def load_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def gemini_client(api_key: Optional[str] = None) -> genai.Client:
    # If GEMINI_API_KEY env var is set, genai.Client() will use it.
    return genai.Client(api_key=api_key) if api_key else genai.Client()


def build_labeled_contents(prompt: str, image_paths: List[str]) -> List[object]:
    """
    Builds a single "contents" list that includes:
      - the user's prompt
      - an ordered index of images
      - per-image labels + image bytes parts
    """
    # A compact index list up front helps the model keep track of ordering.
    index_lines = ["You will receive images in this exact order:"]
    for i, p in enumerate(image_paths):
        index_lines.append(f"{i}: {os.path.basename(p)}")
    index_block = "\n".join(index_lines)

    contents: List[object] = [prompt.strip(), "\n" + index_block + "\n"]

    for i, p in enumerate(image_paths):
        label = f"\nImage {i} ({os.path.basename(p)}):"
        contents.append(label)
        contents.append(types.Part.from_bytes(data=load_bytes(p), mime_type="image/png"))

    return contents


def send_all_images_to_gemini_one_prompt(
    client: genai.Client,
    model: str,
    prompt: str,
    image_paths: List[str],
):
    contents = build_labeled_contents(prompt, image_paths)
    resp = client.models.generate_content(model=model, contents=contents)
    return resp


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb", help="Path to .glb asset pack")
    ap.add_argument("--out", default="glb_renders", help="Output directory")

    # Slightly smaller than 1024, still > 768:
    ap.add_argument("--w", type=int, default=896, help="Image width (default 896)")
    ap.add_argument("--h", type=int, default=896, help="Image height (default 896)")

    ap.add_argument(
        "--views",
        default="45,20;225,20;45,60;225,60",
        help='Semicolon-separated "az,el" degree pairs. Example: "45,20;225,20"',
    )
    ap.add_argument("--engine", default="EEVEE", choices=["EEVEE", "CYCLES"])
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--bg-strength", type=float, default=0.6)

    ap.add_argument("--transparent", dest="transparent", action="store_true")
    ap.add_argument("--no-transparent", dest="transparent", action="store_false")
    ap.set_defaults(transparent=True)

    ap.add_argument(
        "--blender",
        default="/Applications/Blender.app/Contents/MacOS/Blender",
        help="Path to Blender executable",
    )

    # Gemini options
    ap.add_argument("--model", default="gemini-3-flash-preview", help="Gemini model name")
    ap.add_argument("--api-key", default=None, help="Optional API key override")
    ap.add_argument(
        "--results",
        default=None,
        help="Output JSONL file (default: <out>/gemini_results.jsonl)",
    )

    args = ap.parse_args()

    if not os.path.exists(args.blender):
        print(f"ERROR: Blender not found at: {args.blender}", file=sys.stderr)
        sys.exit(2)

    # 1) Render
    run_blender_render(
        blender_path=args.blender,
        glb_path=args.glb,
        out_dir=args.out,
        width=args.w,
        height=args.h,
        views=args.views,
        engine=args.engine,
        transparent=args.transparent,
        bg_strength=args.bg_strength,
        samples=args.samples,
    )

    pngs = list_rendered_pngs(args.out)
    if not pngs:
        print("ERROR: No rendered PNGs found.", file=sys.stderr)
        sys.exit(3)

    # 2) Send ONE prompt with ALL images
    client = gemini_client(api_key=args.api_key)
    print(f"Sending {len(pngs)} images to Gemini in one request...")
    resp = send_all_images_to_gemini_one_prompt(
        client=client,
        model=args.model,
        prompt=GEMINI_PROMPT,
        image_paths=pngs,
    )

    # 3) Save one combined record
    results_path = args.results or os.path.join(args.out, "gemini_results.jsonl")
    raw = resp.model_dump(mode="json") if hasattr(resp, "model_dump") else None

    record = {
        "glb_path": os.path.abspath(args.glb),
        "render_dir": os.path.abspath(args.out),
        "image_paths": [os.path.abspath(p) for p in pngs],
        "model": args.model,
        "prompt": GEMINI_PROMPT,
        "response_text": getattr(resp, "text", None),
        "raw": raw,
        "ts": time.time(),
    }

    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. Wrote combined Gemini response to: {results_path}")


if __name__ == "__main__":
    main()
