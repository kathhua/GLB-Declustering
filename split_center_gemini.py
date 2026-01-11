# # #!/usr/bin/env python3
# # """
# # Split an asset-pack GLB (e.g. 10 cars in a circle) into separate GLB files,
# # one per object — with eps auto-tuned to match Gemini's predicted object_count.

# # What changed vs your original script?
# # -------------------------------------
# # - Added ONE function: `auto_eps_split_glb(...)`

# # That function:
# # 1) Reads Gemini's JSON/JSONL output to get the target object_count
# # 2) Sweeps eps from eps_min .. eps_max in steps (default 0.05)
# # 3) Runs clustering (WITHOUT exporting files) to estimate #clusters
# # 4) When it finds eps that yields EXACTLY the target #clusters, it runs the real split/export once.

# # So you do NOT need to pass --eps manually anymore.

# # Usage
# # -----
# # 1) Ensure your Gemini results JSONL contains something like:
# #    {"response_text": "...", "raw": ..., ...}
# #    OR your prompt returns JSON and you stored it in response_text like:
# #    {"object_count": 10, ...}

# # 2) Run:
# #    python split_glb_auto_eps.py input.glb --gemini-jsonl renders/gemini_results.jsonl -o split_objects

# # Notes
# # -----
# # - This assumes your Gemini output includes a numeric "object_count" somewhere in either:
# #   - response_text (as JSON), or
# #   - raw (as structured model output), or
# #   - any line as a dict field named "object_count".
# # - eps search bounds are computed automatically from the centers cloud scale.
# # """

# # import argparse
# # import json
# # import os
# # import re
# # from collections import defaultdict
# # from typing import Any, Dict, List, Optional, Tuple

# # import numpy as np
# # import trimesh
# # from sklearn.cluster import DBSCAN


# # def extract_world_meshes(scene: trimesh.Scene) -> Tuple[List[trimesh.Trimesh], np.ndarray]:
# #     """
# #     From a trimesh.Scene, return:

# #     meshes_world: [Trimesh, ...] each already transformed to world space
# #     centers:      (N, 3) array of bounding-box centers for each mesh
# #     """
# #     meshes_world: List[trimesh.Trimesh] = []
# #     centers: List[np.ndarray] = []

# #     for node_name in scene.graph.nodes_geometry:
# #         matrix, geom_name = scene.graph.get(frame_to=node_name)
# #         if geom_name is None:
# #             continue

# #         base_mesh = scene.geometry[geom_name]
# #         mesh = base_mesh.copy()
# #         mesh.apply_transform(matrix)

# #         meshes_world.append(mesh)
# #         centers.append(mesh.bounding_box.centroid)

# #     centers_arr = np.vstack(centers) if centers else np.zeros((0, 3), dtype=np.float32)
# #     return meshes_world, centers_arr


# # def cluster_meshes(centers: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
# #     """
# #     Cluster meshes based on their centers using DBSCAN.

# #     Returns:
# #         labels: array of cluster labels for each center (shape (N,))
# #                 -1 means "noise" (not part of any cluster).
# #     """
# #     if len(centers) == 0:
# #         return np.array([], dtype=int)

# #     db = DBSCAN(eps=eps, min_samples=min_samples)
# #     labels = db.fit_predict(centers)
# #     return labels


# # def split_glb(
# #     input_glb: str,
# #     out_dir: str,
# #     eps: float,
# #     min_samples: int = 1,
# # ) -> int:
# #     """
# #     Split and export per-object GLBs using the given eps.
# #     Returns the number of clusters written.
# #     """
# #     scene = trimesh.load(input_glb, force="scene")

# #     meshes_world, centers = extract_world_meshes(scene)
# #     if not meshes_world:
# #         raise RuntimeError("No geometry nodes found in the GLB.")

# #     labels = cluster_meshes(centers, eps=eps, min_samples=min_samples)

# #     groups: Dict[int, List[trimesh.Trimesh]] = defaultdict(list)
# #     next_noise_label = (int(labels.max()) + 1) if len(labels) else 0

# #     for mesh, label in zip(meshes_world, labels):
# #         if label == -1:
# #             label = next_noise_label
# #             next_noise_label += 1
# #         groups[int(label)].append(mesh)

# #     base_name = os.path.splitext(os.path.basename(input_glb))[0]
# #     os.makedirs(out_dir, exist_ok=True)

# #     print(f"[split_glb] eps={eps:.3f} -> Found {len(groups)} object clusters")

# #     for i, (_, mesh_list) in enumerate(sorted(groups.items()), start=1):
# #         combined = mesh_list[0] if len(mesh_list) == 1 else trimesh.util.concatenate(mesh_list)
# #         out_path = os.path.join(out_dir, f"{base_name}_object_{i:02d}.glb")
# #         combined.export(out_path)
# #         print(f"  -> wrote {out_path}")

# #     return len(groups)


# # # ======================================================================
# # # ADDED FUNCTION (the only “new feature” you requested)
# # # ======================================================================
# # def auto_eps_split_glb(
# #     input_glb: str,
# #     out_dir: str,
# #     gemini_jsonl_path: str,
# #     eps_step: float = 0.05,
# #     min_samples: int = 1,
# #     eps_min: Optional[float] = None,
# #     eps_max: Optional[float] = None,
# # ) -> Tuple[float, int]:
# #     """
# #     Iterates eps values until the estimated cluster count == Gemini's object_count.
# #     Exports per-object GLBs ONCE using the first eps that matches.

# #     Returns:
# #         (best_eps, cluster_count)

# #     If no eps matches exactly, raises RuntimeError (and prints best attempts).
# #     """

# #     def _read_jsonl_last(path: str) -> Dict[str, Any]:
# #         last: Optional[Dict[str, Any]] = None
# #         with open(path, "r", encoding="utf-8") as f:
# #             for line in f:
# #                 line = line.strip()
# #                 if not line:
# #                     continue
# #                 try:
# #                     last = json.loads(line)
# #                 except json.JSONDecodeError:
# #                     continue
# #         if last is None:
# #             raise RuntimeError(f"Gemini JSONL is empty or invalid: {path}")
# #         return last

# #     def _extract_object_count(obj: Any) -> Optional[int]:
# #         """
# #         Try hard to find an integer object_count inside:
# #         - dict keys
# #         - nested dict/list
# #         - JSON string in response_text
# #         - fallback regex inside response_text
# #         """
# #         if obj is None:
# #             return None

# #         if isinstance(obj, int):
# #             return obj

# #         if isinstance(obj, dict):
# #             # direct hit
# #             for k in ("object_count", "objectCount", "count", "num_objects", "numObjects"):
# #                 if k in obj and isinstance(obj[k], (int, float, str)):
# #                     try:
# #                         return int(float(obj[k]))
# #                     except Exception:
# #                         pass

# #             # response_text might be JSON
# #             if "response_text" in obj and isinstance(obj["response_text"], str):
# #                 s = obj["response_text"].strip()
# #                 # try parse as JSON first
# #                 try:
# #                     parsed = json.loads(s)
# #                     got = _extract_object_count(parsed)
# #                     if got is not None:
# #                         return got
# #                 except Exception:
# #                     pass
# #                 # regex fallback: object_count: 12  OR  "object_count": 12
# #                 m = re.search(r'object_count"\s*:\s*(\d+)', s)
# #                 if m:
# #                     return int(m.group(1))
# #                 m = re.search(r"\bobject_count\b\s*[:=]\s*(\d+)", s)
# #                 if m:
# #                     return int(m.group(1))

# #             # search nested
# #             for v in obj.values():
# #                 got = _extract_object_count(v)
# #                 if got is not None:
# #                     return got
# #             return None

# #         if isinstance(obj, list):
# #             for v in obj:
# #                 got = _extract_object_count(v)
# #                 if got is not None:
# #                     return got
# #             return None

# #         if isinstance(obj, str):
# #             # maybe the whole thing is JSON
# #             s = obj.strip()
# #             try:
# #                 parsed = json.loads(s)
# #                 return _extract_object_count(parsed)
# #             except Exception:
# #                 pass
# #             m = re.search(r'object_count"\s*:\s*(\d+)', s)
# #             if m:
# #                 return int(m.group(1))
# #             m = re.search(r"\bobject_count\b\s*[:=]\s*(\d+)", s)
# #             if m:
# #                 return int(m.group(1))
# #             return None

# #         return None

# #     # 1) Read Gemini target
# #     last_record = _read_jsonl_last(gemini_jsonl_path)
# #     target = _extract_object_count(last_record)
# #     if target is None:
# #         raise RuntimeError(
# #             "Could not find an integer object_count in the Gemini JSONL record. "
# #             "Make sure your prompt returns JSON with an object_count field."
# #         )
# #     if target <= 0:
# #         raise RuntimeError(f"Gemini target object_count is invalid: {target}")

# #     print(f"[auto_eps] Gemini target object_count = {target}")

# #     # 2) Load GLB once & compute centers scale
# #     scene = trimesh.load(input_glb, force="scene")
# #     meshes_world, centers = extract_world_meshes(scene)
# #     if not meshes_world or len(centers) == 0:
# #         raise RuntimeError("No geometry nodes found in the GLB (cannot auto-tune eps).")

# #     # Compute a reasonable eps sweep range from centers cloud
# #     cmin = centers.min(axis=0)
# #     cmax = centers.max(axis=0)
# #     diag = float(np.linalg.norm(cmax - cmin))
# #     diag = max(diag, 1e-6)

# #     # Defaults:
# #     # - eps_min: small but not zero (too small makes every mesh its own cluster)
# #     # - eps_max: big enough to merge everything (upper bound)
# #     if eps_min is None:
# #         eps_min = max(0.01, diag * 0.005)  # small fraction of scene scale
# #     if eps_max is None:
# #         eps_max = diag * 1.25  # enough to merge across the whole spread

# #     # Round bounds for clean stepping
# #     eps = float(eps_min)
# #     eps_max = float(eps_max)
# #     eps_step = float(eps_step)

# #     # 3) Sweep eps and estimate #clusters (no export)
# #     tried: List[Tuple[float, int]] = []
# #     best_close: Optional[Tuple[float, int, int]] = None  # (eps, clusters, abs_diff)

# #     def estimate_clusters_for_eps(eps_val: float) -> int:
# #         labels = cluster_meshes(centers, eps=eps_val, min_samples=min_samples)
# #         groups = set()
# #         next_noise_label = (int(labels.max()) + 1) if len(labels) else 0
# #         for lab in labels:
# #             if int(lab) == -1:
# #                 groups.add(next_noise_label)
# #                 next_noise_label += 1
# #             else:
# #                 groups.add(int(lab))
# #         return len(groups)

# #     # Make sure we don't loop forever due to float drift
# #     max_iters = int(np.ceil((eps_max - eps_min) / eps_step)) + 2

# #     print(f"[auto_eps] Searching eps in [{eps_min:.3f}, {eps_max:.3f}] step={eps_step:.3f} (max {max_iters} iters)")

# #     for _ in range(max_iters):
# #         if eps > eps_max + 1e-9:
# #             break

# #         n_clusters = estimate_clusters_for_eps(eps)
# #         tried.append((eps, n_clusters))

# #         diff = abs(n_clusters - target)
# #         if best_close is None or diff < best_close[2]:
# #             best_close = (eps, n_clusters, diff)

# #         # Exact match => export ONCE and return
# #         if n_clusters == target:
# #             print(f"[auto_eps] Found exact match: eps={eps:.3f} -> clusters={n_clusters}")
# #             # Now do the real split/export with this eps
# #             written = split_glb(input_glb, out_dir, eps=eps, min_samples=min_samples)
# #             return eps, written

# #         eps = round(eps + eps_step, 10)

# #     # 4) If no exact match, show best attempts and error out
# #     if best_close is not None:
# #         beps, bclusters, bdiff = best_close
# #         print(f"[auto_eps] No exact eps match found.")
# #         print(f"[auto_eps] Closest: eps={beps:.3f} -> clusters={bclusters} (diff={bdiff})")

# #         # Print a few nearest tries around the best eps (for debugging)
# #         tried_sorted = sorted(tried, key=lambda x: abs(x[1] - target))
# #         print("[auto_eps] Top 10 closest eps values:")
# #         for e, c in tried_sorted[:10]:
# #             print(f"  eps={e:.3f} -> clusters={c}")

# #     raise RuntimeError("Could not find an eps value that yields exactly Gemini's object_count.")


# # def main():
# #     parser = argparse.ArgumentParser(
# #         description="Split a GLB asset pack into individual GLB files with eps auto-tuned to Gemini's object_count."
# #     )
# #     parser.add_argument("input_glb", help="Path to input .glb file (e.g. cars.glb)")
# #     parser.add_argument(
# #         "-o",
# #         "--outdir",
# #         default="split_objects",
# #         help="Output directory for per-object GLBs",
# #     )
# #     parser.add_argument(
# #         "--gemini-jsonl",
# #         required=True,
# #         help="Path to the Gemini results JSONL produced by your render+Gemini script.",
# #     )
# #     parser.add_argument(
# #         "--eps-step",
# #         type=float,
# #         default=0.05,
# #         help="eps sweep step size (0.05 recommended).",
# #     )
# #     parser.add_argument(
# #         "--min-samples",
# #         type=int,
# #         default=1,
# #         help="DBSCAN min_samples; 1 works well for this use-case.",
# #     )
# #     # Optional overrides if you want to cap the search region
# #     parser.add_argument("--eps-min", type=float, default=None, help="Optional eps minimum for search.")
# #     parser.add_argument("--eps-max", type=float, default=None, help="Optional eps maximum for search.")

# #     args = parser.parse_args()

# #     best_eps, clusters = auto_eps_split_glb(
# #         input_glb=args.input_glb,
# #         out_dir=args.outdir,
# #         gemini_jsonl_path=args.gemini_jsonl,
# #         eps_step=args.eps_step,
# #         min_samples=args.min_samples,
# #         eps_min=args.eps_min,
# #         eps_max=args.eps_max,
# #     )

# #     print(f"\nDone. Auto-selected eps={best_eps:.3f}; wrote {clusters} object GLBs to: {args.outdir}")


# # if __name__ == "__main__":
# #     main()


# #!/usr/bin/env python3
# """
# glb_render_gemini_then_split.py

# END-TO-END PIPELINE (single script):
# 1) Render a .glb asset pack to multiple PNG images using Blender (headless).
# 2) Send ALL rendered images to Gemini in ONE request with ONE prompt (prompt is hardcoded below).
#    - Images are labeled in the prompt so Gemini can reference them consistently.
# 3) Append the Gemini response to a JSONL file.
# 4) Read the *correct* record back from the JSONL (NOT “latest line” blindly):
#    - This script writes a unique run_id into the record, then re-reads the JSONL and finds
#      the line with that exact run_id.
# 5) Extract Gemini’s predicted object count.
# 6) Auto-search DBSCAN eps until the split cluster count matches Gemini’s object count exactly.
# 7) Export per-object GLB files using ONLY that matched eps.

# Requirements:
#   - Blender installed
#   - pip install -U google-genai trimesh numpy scikit-learn

# Auth:
#   - export GEMINI_API_KEY="..."

# Usage:
#   python glb_render_gemini_then_split.py input.glb --out gemini_runs --split-out split_objects

# Notes:
# - Your Gemini prompt currently asks for ONLY the total count with no explanation.
#   This script supports that (parses the first integer it finds).
# - If you later change Gemini to return JSON, it will also work (it tries JSON parsing too).
# """

# import argparse
# import glob
# import json
# import os
# import re
# import subprocess
# import sys
# import tempfile
# import time
# import uuid
# from collections import defaultdict
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np
# import trimesh
# from sklearn.cluster import DBSCAN

# from google import genai
# from google.genai import types


# # ============================================================
# # 0) HARD-CODE YOUR GEMINI PROMPT HERE (included exactly)
# # ============================================================
# GEMINI_PROMPT = (
#     "Given these images of a GLB 3D asset pack, carefully identify the number of distinct objects present "
#     "in the pack. An object is defined as a separate, individual item that can be distinguished from others "
#     "based on its geometry and structure. Consider factors such as connectivity, spatial separation, and unique "
#     "features when determining object boundaries. Provide only the total count of distinct objects without any "
#     "additional explanation."
# )


# # ============================================================
# # 1) Blender rendering script (embedded)
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
# width    = int(args.get("w", "896"))
# height   = int(args.get("h", "896"))
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
# # 2) Blender render helpers
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
#     return sorted(glob.glob(os.path.join(out_dir, "view_*.png")))


# def load_bytes(path: str) -> bytes:
#     with open(path, "rb") as f:
#         return f.read()


# # ============================================================
# # 3) Gemini helpers (ONE prompt + MANY images, labeled)
# # ============================================================
# def gemini_client(api_key: Optional[str] = None) -> genai.Client:
#     return genai.Client(api_key=api_key) if api_key else genai.Client()


# def build_labeled_contents(prompt: str, image_paths: List[str]) -> List[object]:
#     index_lines = ["You will receive images in this exact order:"]
#     for i, p in enumerate(image_paths):
#         index_lines.append(f"{i}: {os.path.basename(p)}")
#     index_block = "\n".join(index_lines)

#     contents: List[object] = [prompt.strip(), "\n" + index_block + "\n"]
#     for i, p in enumerate(image_paths):
#         contents.append(f"\nImage {i} ({os.path.basename(p)}):")
#         contents.append(types.Part.from_bytes(data=load_bytes(p), mime_type="image/png"))
#     return contents


# def send_all_images_to_gemini_one_prompt(
#     client: genai.Client,
#     model: str,
#     prompt: str,
#     image_paths: List[str],
# ):
#     contents = build_labeled_contents(prompt, image_paths)
#     return client.models.generate_content(model=model, contents=contents)


# def append_record_jsonl(path: str, record: Dict[str, Any]) -> None:
#     os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
#     with open(path, "a", encoding="utf-8") as f:
#         f.write(json.dumps(record, ensure_ascii=False) + "\n")


# def find_record_by_run_id(path: str, run_id: str) -> Dict[str, Any]:
#     """
#     Reads JSONL and returns the record with matching run_id.
#     This avoids “just take the latest line”.
#     """
#     if not os.path.exists(path):
#         raise RuntimeError(f"Results JSONL does not exist: {path}")

#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 obj = json.loads(line)
#             except json.JSONDecodeError:
#                 continue
#             if obj.get("run_id") == run_id:
#                 return obj

#     raise RuntimeError(f"Could not find run_id={run_id} in {path}")


# def extract_object_count_from_record(record: Dict[str, Any]) -> int:
#     """
#     Your prompt asks Gemini to output ONLY the total count (like '10').
#     We support:
#       - pure integer text
#       - text with an integer somewhere
#       - JSON with object_count / count fields
#       - nested raw outputs containing object_count
#     """
#     def _extract(obj: Any) -> Optional[int]:
#         if obj is None:
#             return None

#         if isinstance(obj, int):
#             return obj

#         if isinstance(obj, float):
#             return int(obj)

#         if isinstance(obj, str):
#             s = obj.strip()

#             # Try JSON first (in case you later change prompt to JSON)
#             try:
#                 parsed = json.loads(s)
#                 got = _extract(parsed)
#                 if got is not None:
#                     return got
#             except Exception:
#                 pass

#             # Otherwise, find the first integer in the string
#             m = re.search(r"\b(\d+)\b", s)
#             if m:
#                 return int(m.group(1))
#             return None

#         if isinstance(obj, dict):
#             # direct keys
#             for k in ("object_count", "objectCount", "count", "num_objects", "numObjects", "total"):
#                 if k in obj:
#                     got = _extract(obj[k])
#                     if got is not None:
#                         return got
#             # search nested
#             for v in obj.values():
#                 got = _extract(v)
#                 if got is not None:
#                     return got
#             return None

#         if isinstance(obj, list):
#             for v in obj:
#                 got = _extract(v)
#                 if got is not None:
#                     return got
#             return None

#         return None

#     # Most likely: record["response_text"] == "10"
#     got = _extract(record.get("response_text"))
#     if got is not None and got > 0:
#         return got

#     # fallback: raw
#     got = _extract(record.get("raw"))
#     if got is not None and got > 0:
#         return got

#     raise RuntimeError(
#         "Could not extract a positive integer object count from Gemini record. "
#         "Check record['response_text'] and record['raw'] in the JSONL."
#     )


# # ============================================================
# # 4) GLB splitting (DBSCAN), with AUTO eps search
# # ============================================================
# def extract_world_meshes(scene: trimesh.Scene) -> Tuple[List[trimesh.Trimesh], np.ndarray]:
#     meshes_world: List[trimesh.Trimesh] = []
#     centers: List[np.ndarray] = []

#     for node_name in scene.graph.nodes_geometry:
#         matrix, geom_name = scene.graph.get(frame_to=node_name)
#         if geom_name is None:
#             continue

#         base_mesh = scene.geometry[geom_name]
#         mesh = base_mesh.copy()
#         mesh.apply_transform(matrix)

#         meshes_world.append(mesh)
#         centers.append(mesh.bounding_box.centroid)

#     centers_arr = np.vstack(centers) if centers else np.zeros((0, 3), dtype=np.float32)
#     return meshes_world, centers_arr


# def cluster_meshes(centers: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
#     if len(centers) == 0:
#         return np.array([], dtype=int)
#     return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(centers)


# def estimate_num_clusters(labels: np.ndarray) -> int:
#     """
#     DBSCAN labels can include -1 as noise. We treat each noise point as its own cluster
#     (same logic as your original splitting approach intended, but correctly per-point).
#     """
#     if labels.size == 0:
#         return 0

#     clusters = set()
#     next_noise_label = int(labels.max()) + 1 if labels.size else 0
#     for lab in labels:
#         lab = int(lab)
#         if lab == -1:
#             clusters.add(next_noise_label)
#             next_noise_label += 1
#         else:
#             clusters.add(lab)
#     return len(clusters)


# def split_glb_with_eps(
#     input_glb: str,
#     out_dir: str,
#     eps: float,
#     min_samples: int = 1,
# ) -> int:
#     scene = trimesh.load(input_glb, force="scene")
#     meshes_world, centers = extract_world_meshes(scene)
#     if not meshes_world:
#         raise RuntimeError("No geometry nodes found in the GLB.")

#     labels = cluster_meshes(centers, eps=eps, min_samples=min_samples)

#     groups: Dict[int, List[trimesh.Trimesh]] = defaultdict(list)
#     next_noise_label = int(labels.max()) + 1 if labels.size else 0

#     for mesh, lab in zip(meshes_world, labels):
#         lab = int(lab)
#         if lab == -1:
#             lab = next_noise_label
#             next_noise_label += 1
#         groups[lab].append(mesh)

#     base_name = os.path.splitext(os.path.basename(input_glb))[0]
#     os.makedirs(out_dir, exist_ok=True)

#     print(f"[split] eps={eps:.3f} -> Found {len(groups)} object clusters (exporting...)")

#     for i, (_, mesh_list) in enumerate(sorted(groups.items()), start=1):
#         combined = mesh_list[0] if len(mesh_list) == 1 else trimesh.util.concatenate(mesh_list)
#         out_path = os.path.join(out_dir, f"{base_name}_object_{i:02d}.glb")
#         combined.export(out_path)
#         print(f"  -> wrote {out_path}")

#     return len(groups)


# def auto_find_eps_matching_target(
#     input_glb: str,
#     target_clusters: int,
#     eps_step: float = 0.025,
#     min_samples: int = 1,
#     eps_min: Optional[float] = None,
#     eps_max: Optional[float] = None,
# ) -> float:
#     scene = trimesh.load(input_glb, force="scene")
#     _, centers = extract_world_meshes(scene)
#     if len(centers) == 0:
#         raise RuntimeError("No geometry nodes found in GLB; cannot auto-tune eps.")

#     cmin = centers.min(axis=0)
#     cmax = centers.max(axis=0)
#     diag = float(np.linalg.norm(cmax - cmin))
#     diag = max(diag, 1e-6)

#     if eps_min is None:
#         eps_min = max(0.01, diag * 0.005)
#     if eps_max is None:
#         eps_max = diag * 1.25

#     max_iters = int(np.ceil((eps_max - eps_min) / eps_step)) + 2

#     best_close: Optional[Tuple[float, int, int]] = None  # eps, clusters, absdiff

#     print(f"[auto-eps] Target clusters={target_clusters}")
#     print(f"[auto-eps] Searching eps in [{eps_min:.3f}, {eps_max:.3f}] step={eps_step:.3f} (max {max_iters} iters)")

#     eps = float(eps_min)
#     for _ in range(max_iters):
#         if eps > eps_max + 1e-9:
#             break

#         labels = cluster_meshes(centers, eps=eps, min_samples=min_samples)
#         n = estimate_num_clusters(labels)

#         diff = abs(n - target_clusters)
#         if best_close is None or diff < best_close[2]:
#             best_close = (eps, n, diff)

#         if n == target_clusters:
#             print(f"[auto-eps] Exact match: eps={eps:.3f} -> clusters={n}")
#             return eps

#         eps = round(eps + eps_step, 10)

#     if best_close:
#         beps, bn, bdiff = best_close
#         raise RuntimeError(
#             f"No eps found that matches target exactly.\n"
#             f"Closest was eps={beps:.3f} -> clusters={bn} (diff={bdiff}).\n"
#             f"Try adjusting eps_step or your geometry/packing."
#         )

#     raise RuntimeError("No eps found (unexpected).")


# # ============================================================
# # 5) Main pipeline
# # ============================================================
# def main():
#     ap = argparse.ArgumentParser(description="Render GLB -> Gemini count -> auto eps -> split GLB into per-object GLBs")

#     ap.add_argument("glb", help="Path to input .glb asset pack")

#     # Rendering outputs
#     ap.add_argument("--out", default="glb_renders", help="Directory to store rendered images + gemini jsonl")
#     ap.add_argument("--w", type=int, default=896, help="Render width (>768). Default 896.")
#     ap.add_argument("--h", type=int, default=896, help="Render height (>768). Default 896.")
#     ap.add_argument("--views", default="45,20;225,20;45,60;225,60",
#                     help='Semicolon-separated "az,el" pairs. Example: "45,20;225,20"')
#     ap.add_argument("--engine", default="EEVEE", choices=["EEVEE", "CYCLES"])
#     ap.add_argument("--samples", type=int, default=64)
#     ap.add_argument("--bg-strength", type=float, default=0.6)
#     ap.add_argument("--transparent", dest="transparent", action="store_true")
#     ap.add_argument("--no-transparent", dest="transparent", action="store_false")
#     ap.set_defaults(transparent=True)

#     ap.add_argument("--blender", default="/Applications/Blender.app/Contents/MacOS/Blender",
#                     help="Path to Blender executable")

#     # Gemini options
#     ap.add_argument("--model", default="gemini-3-flash-preview", help="Gemini model name")
#     ap.add_argument("--api-key", default=None, help="Optional API key override (else GEMINI_API_KEY env var)")
#     ap.add_argument("--results", default=None, help="Results JSONL (default: <out>/gemini_results.jsonl)")

#     # Splitting outputs
#     ap.add_argument("--split-out", default="split_objects", help="Output dir for per-object GLBs")

#     # eps auto-search tuning
#     ap.add_argument("--eps-step", type=float, default=0.05, help="eps sweep step size")
#     ap.add_argument("--min-samples", type=int, default=1, help="DBSCAN min_samples (default 1)")
#     ap.add_argument("--eps-min", type=float, default=None, help="Optional eps min override")
#     ap.add_argument("--eps-max", type=float, default=None, help="Optional eps max override")

#     args = ap.parse_args()

#     if not os.path.exists(args.blender):
#         print(f"ERROR: Blender not found at: {args.blender}", file=sys.stderr)
#         sys.exit(2)

#     glb_abs = os.path.abspath(args.glb)
#     out_dir = os.path.abspath(args.out)
#     os.makedirs(out_dir, exist_ok=True)

#     # ---- Step 1: Render ----
#     run_blender_render(
#         blender_path=args.blender,
#         glb_path=glb_abs,
#         out_dir=out_dir,
#         width=args.w,
#         height=args.h,
#         views=args.views,
#         engine=args.engine,
#         transparent=args.transparent,
#         bg_strength=args.bg_strength,
#         samples=args.samples,
#     )

#     pngs = list_rendered_pngs(out_dir)
#     if not pngs:
#         print("ERROR: No rendered PNGs found.", file=sys.stderr)
#         sys.exit(3)

#     # ---- Step 2: Send to Gemini (ONE request) ----
#     client = gemini_client(api_key=args.api_key)
#     run_id = str(uuid.uuid4())
#     results_path = args.results or os.path.join(out_dir, "gemini_results.jsonl")

#     print(f"Sending {len(pngs)} images to Gemini in one request...")
#     resp = send_all_images_to_gemini_one_prompt(
#         client=client,
#         model=args.model,
#         prompt=GEMINI_PROMPT,
#         image_paths=pngs,
#     )

#     raw = resp.model_dump(mode="json") if hasattr(resp, "model_dump") else None
#     record = {
#         "run_id": run_id,
#         "glb_path": glb_abs,
#         "render_dir": out_dir,
#         "image_paths": [os.path.abspath(p) for p in pngs],
#         "model": args.model,
#         "prompt": GEMINI_PROMPT,
#         "response_text": getattr(resp, "text", None),
#         "raw": raw,
#         "ts": time.time(),
#     }
#     append_record_jsonl(results_path, record)
#     print(f"Gemini response appended to: {results_path}")
#     print(f"run_id: {run_id}")

#     # ---- Step 3: Read the correct output back from JSONL ----
#     record_from_file = find_record_by_run_id(results_path, run_id)
#     target_count = extract_object_count_from_record(record_from_file)
#     print(f"[gemini] Parsed target object_count = {target_count}")

#     # ---- Step 4: Auto-find eps to match target_count ----
#     best_eps = auto_find_eps_matching_target(
#         input_glb=glb_abs,
#         target_clusters=target_count,
#         eps_step=args.eps_step,
#         min_samples=args.min_samples,
#         eps_min=args.eps_min,
#         eps_max=args.eps_max,
#     )

#     # ---- Step 5: Split/export once with best_eps ----
#     split_out = os.path.abspath(args.split_out)
#     written = split_glb_with_eps(
#         input_glb=glb_abs,
#         out_dir=split_out,
#         eps=best_eps,
#         min_samples=args.min_samples,
#     )

#     print("\n✅ Done.")
#     print(f"Gemini target count: {target_count}")
#     print(f"Selected eps: {best_eps:.3f}")
#     print(f"Wrote {written} GLB files to: {split_out}")
#     print(f"Gemini JSONL: {results_path}")
#     print(f"run_id: {run_id}")


# if __name__ == "__main__":
#     main()




#!/usr/bin/env python3
"""
glb_render_gemini_then_split.py

END-TO-END PIPELINE (single script):
1) Render a .glb asset pack to multiple PNG images using Blender (headless).
2) Send ALL rendered images to Gemini in ONE request with ONE prompt (prompt hardcoded below).
   - Images are labeled in the prompt.
3) Append the Gemini response to a JSONL file.
4) Read the *correct* record back from the JSONL (NOT “latest line” blindly):
   - This script writes a unique run_id into the record, then re-reads the JSONL and finds
     the line with that exact run_id.
5) Extract Gemini’s predicted object count.
6) Auto-search DBSCAN eps for an EXACT match to Gemini’s object count.
   - If no exact eps exists, it will:
       a) report it couldn't find a perfect match
       b) choose the closest eps (minimizing |clusters - target|)
       c) STILL split/export using that closest eps.
7) Export per-object GLB files using the selected eps.

Requirements:
  - Blender installed
  - pip install -U google-genai trimesh numpy scikit-learn

Auth:
  - export GEMINI_API_KEY="..."

Usage:
  python glb_render_gemini_then_split.py input.glb --out gemini_runs --split-out split_objects
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh
from sklearn.cluster import DBSCAN

from google import genai
from google.genai import types


# ============================================================
# 0) HARD-CODE YOUR GEMINI PROMPT HERE (included exactly)
# ============================================================
GEMINI_PROMPT = (
    "Given these images of a GLB 3D asset pack, carefully identify the number of distinct objects present in the pack. "
    "An object is defined as a separate, individual item that can be distinguished from others based on its geometry and structure. "
    "Consider factors such as connectivity, spatial separation, and unique features when determining object boundaries. "
    "Provide only the total count of distinct objects without any additional explanation."
)


# ============================================================
# 1) Blender rendering script (embedded)
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

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb_path)

mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not mesh_objs:
    raise RuntimeError("No mesh objects found after importing GLB.")

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

cam_data = bpy.data.cameras.new("Camera")
cam = bpy.data.objects.new("Camera", cam_data)
bpy.context.scene.collection.objects.link(cam)
bpy.context.scene.camera = cam

cam.data.lens_unit = 'FOV'
cam.data.angle = math.radians(45.0)

target = bpy.data.objects.new("Target", None)
target.empty_display_type = 'PLAIN_AXES'
target.empty_display_size = max(radius * 0.05, 0.01)
target.location = center
bpy.context.scene.collection.objects.link(target)

track = cam.constraints.new(type='TRACK_TO')
track.target = target
track.track_axis = 'TRACK_NEGATIVE_Z'
track.up_axis = 'UP_Y'

world = bpy.data.worlds.new("World")
world.use_nodes = True
bpy.context.scene.world = world
bg = world.node_tree.nodes.get("Background")
bg.inputs[1].default_value = bg_strength

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

def set_camera_view(az_deg, el_deg):
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    fov = cam.data.angle
    dist = radius / math.sin(fov * 0.5)
    dist *= 1.25

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
# 2) Blender helpers
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


def load_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


# ============================================================
# 3) Gemini helpers
# ============================================================
def gemini_client(api_key: Optional[str] = None) -> genai.Client:
    return genai.Client(api_key=api_key) if api_key else genai.Client()


def build_labeled_contents(prompt: str, image_paths: List[str]) -> List[object]:
    index_lines = ["You will receive images in this exact order:"]
    for i, p in enumerate(image_paths):
        index_lines.append(f"{i}: {os.path.basename(p)}")
    index_block = "\n".join(index_lines)

    contents: List[object] = [prompt.strip(), "\n" + index_block + "\n"]
    for i, p in enumerate(image_paths):
        contents.append(f"\nImage {i} ({os.path.basename(p)}):")
        contents.append(types.Part.from_bytes(data=load_bytes(p), mime_type="image/png"))
    return contents


def send_all_images_to_gemini_one_prompt(
    client: genai.Client,
    model: str,
    prompt: str,
    image_paths: List[str],
):
    contents = build_labeled_contents(prompt, image_paths)
    return client.models.generate_content(model=model, contents=contents)


def append_record_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_record_by_run_id(path: str, run_id: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise RuntimeError(f"Results JSONL does not exist: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("run_id") == run_id:
                return obj
    raise RuntimeError(f"Could not find run_id={run_id} in {path}")


def extract_object_count_from_record(record: Dict[str, Any]) -> int:
    def _extract(obj: Any) -> Optional[int]:
        if obj is None:
            return None
        if isinstance(obj, int):
            return obj
        if isinstance(obj, float):
            return int(obj)
        if isinstance(obj, str):
            s = obj.strip()
            # If it’s JSON, parse it
            try:
                parsed = json.loads(s)
                got = _extract(parsed)
                if got is not None:
                    return got
            except Exception:
                pass
            # Otherwise take first integer token
            m = re.search(r"\b(\d+)\b", s)
            return int(m.group(1)) if m else None
        if isinstance(obj, dict):
            for k in ("object_count", "objectCount", "count", "num_objects", "numObjects", "total"):
                if k in obj:
                    got = _extract(obj[k])
                    if got is not None:
                        return got
            for v in obj.values():
                got = _extract(v)
                if got is not None:
                    return got
            return None
        if isinstance(obj, list):
            for v in obj:
                got = _extract(v)
                if got is not None:
                    return got
            return None
        return None

    got = _extract(record.get("response_text"))
    if got is not None and got > 0:
        return got

    got = _extract(record.get("raw"))
    if got is not None and got > 0:
        return got

    raise RuntimeError("Could not extract a positive integer object count from Gemini record.")


# ============================================================
# 4) GLB splitting + eps search (exact-or-closest)
# ============================================================
def extract_world_meshes(scene: trimesh.Scene) -> Tuple[List[trimesh.Trimesh], np.ndarray]:
    meshes_world: List[trimesh.Trimesh] = []
    centers: List[np.ndarray] = []

    for node_name in scene.graph.nodes_geometry:
        matrix, geom_name = scene.graph.get(frame_to=node_name)
        if geom_name is None:
            continue
        base_mesh = scene.geometry[geom_name]
        mesh = base_mesh.copy()
        mesh.apply_transform(matrix)
        meshes_world.append(mesh)
        centers.append(mesh.bounding_box.centroid)

    centers_arr = np.vstack(centers) if centers else np.zeros((0, 3), dtype=np.float32)
    return meshes_world, centers_arr


def cluster_meshes(centers: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    if len(centers) == 0:
        return np.array([], dtype=int)
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(centers)


def estimate_num_clusters(labels: np.ndarray) -> int:
    if labels.size == 0:
        return 0
    clusters = set()
    next_noise_label = int(labels.max()) + 1 if labels.size else 0
    for lab in labels:
        lab = int(lab)
        if lab == -1:
            clusters.add(next_noise_label)
            next_noise_label += 1
        else:
            clusters.add(lab)
    return len(clusters)


def split_glb_with_eps(input_glb: str, out_dir: str, eps: float, min_samples: int = 1) -> int:
    scene = trimesh.load(input_glb, force="scene")
    meshes_world, centers = extract_world_meshes(scene)
    if not meshes_world:
        raise RuntimeError("No geometry nodes found in the GLB.")

    labels = cluster_meshes(centers, eps=eps, min_samples=min_samples)

    groups: Dict[int, List[trimesh.Trimesh]] = defaultdict(list)
    next_noise_label = int(labels.max()) + 1 if labels.size else 0

    for mesh, lab in zip(meshes_world, labels):
        lab = int(lab)
        if lab == -1:
            lab = next_noise_label
            next_noise_label += 1
        groups[lab].append(mesh)

    base_name = os.path.splitext(os.path.basename(input_glb))[0]
    os.makedirs(out_dir, exist_ok=True)

    print(f"[split] eps={eps:.3f} -> clusters={len(groups)} (exporting...)")
    for i, (_, mesh_list) in enumerate(sorted(groups.items()), start=1):
        combined = mesh_list[0] if len(mesh_list) == 1 else trimesh.util.concatenate(mesh_list)
        out_path = os.path.join(out_dir, f"{base_name}_object_{i:02d}.glb")
        combined.export(out_path)
        print(f"  -> wrote {out_path}")
    return len(groups)


def find_eps_exact_or_closest(
    input_glb: str,
    target_clusters: int,
    eps_step: float = 0.01,
    min_samples: int = 1,
    eps_min: Optional[float] = None,
    eps_max: Optional[float] = None,
) -> Tuple[float, int, bool, List[Tuple[float, int]]]:
    """
    Returns:
      best_eps,
      best_clusters,
      exact_match (bool),
      top_candidates (list of (eps, clusters) sorted by closeness)
    """
    scene = trimesh.load(input_glb, force="scene")
    _, centers = extract_world_meshes(scene)
    if len(centers) == 0:
        raise RuntimeError("No geometry nodes found in GLB; cannot auto-tune eps.")

    cmin = centers.min(axis=0)
    cmax = centers.max(axis=0)
    diag = float(np.linalg.norm(cmax - cmin))
    diag = max(diag, 1e-6)

    if eps_min is None:
        eps_min = max(0.01, diag * 0.005)
    if eps_max is None:
        eps_max = diag * 1.25

    max_iters = int(np.ceil((eps_max - eps_min) / eps_step)) + 2
    tried: List[Tuple[float, int]] = []

    best_eps = float(eps_min)
    best_clusters = None  # type: ignore[assignment]
    best_diff = 10**18
    exact = False

    print(f"[auto-eps] Target={target_clusters}")
    print(f"[auto-eps] Searching eps in [{eps_min:.3f}, {eps_max:.3f}] step={eps_step:.3f} (max {max_iters} iters)")

    eps = float(eps_min)
    for _ in range(max_iters):
        if eps > eps_max + 1e-9:
            break

        labels = cluster_meshes(centers, eps=eps, min_samples=min_samples)
        n = estimate_num_clusters(labels)
        tried.append((eps, n))

        diff = abs(n - target_clusters)
        if diff < best_diff:
            best_diff = diff
            best_eps = eps
            best_clusters = n

        if n == target_clusters:
            exact = True
            best_eps = eps
            best_clusters = n
            print(f"[auto-eps] Exact match found: eps={eps:.3f} -> clusters={n}")
            break

        eps = round(eps + eps_step, 10)

    # Sort candidates by closeness (and then prefer smaller eps when tied)
    top_candidates = sorted(tried, key=lambda x: (abs(x[1] - target_clusters), x[0]))[:10]
    return best_eps, int(best_clusters), exact, top_candidates


# ============================================================
# 5) Main pipeline
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Render GLB -> Gemini count -> auto eps -> split GLB into per-object GLBs")

    ap.add_argument("glb", help="Path to input .glb asset pack")

    ap.add_argument("--out", default="glb_renders", help="Directory to store rendered images + gemini jsonl")
    ap.add_argument("--w", type=int, default=896, help="Render width (>768). Default 896.")
    ap.add_argument("--h", type=int, default=896, help="Render height (>768). Default 896.")
    ap.add_argument("--views", default="45,20;225,20;45,60;225,60",
                    help='Semicolon-separated "az,el" pairs. Example: "45,20;225,20"')
    ap.add_argument("--engine", default="EEVEE", choices=["EEVEE", "CYCLES"])
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--bg-strength", type=float, default=0.6)
    ap.add_argument("--transparent", dest="transparent", action="store_true")
    ap.add_argument("--no-transparent", dest="transparent", action="store_false")
    ap.set_defaults(transparent=True)

    ap.add_argument("--blender", default="/Applications/Blender.app/Contents/MacOS/Blender",
                    help="Path to Blender executable")

    ap.add_argument("--model", default="gemini-3-flash-preview", help="Gemini model name")
    ap.add_argument("--api-key", default=None, help="Optional API key override (else GEMINI_API_KEY env var)")
    ap.add_argument("--results", default=None, help="Results JSONL (default: <out>/gemini_results.jsonl)")

    ap.add_argument("--split-out", default="split_objects", help="Output dir for per-object GLBs")

    ap.add_argument("--eps-step", type=float, default=0.05, help="eps sweep step size")
    ap.add_argument("--min-samples", type=int, default=1, help="DBSCAN min_samples (default 1)")
    ap.add_argument("--eps-min", type=float, default=None, help="Optional eps min override")
    ap.add_argument("--eps-max", type=float, default=None, help="Optional eps max override")

    args = ap.parse_args()

    if not os.path.exists(args.blender):
        print(f"ERROR: Blender not found at: {args.blender}", file=sys.stderr)
        sys.exit(2)

    glb_abs = os.path.abspath(args.glb)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Render
    run_blender_render(
        blender_path=args.blender,
        glb_path=glb_abs,
        out_dir=out_dir,
        width=args.w,
        height=args.h,
        views=args.views,
        engine=args.engine,
        transparent=args.transparent,
        bg_strength=args.bg_strength,
        samples=args.samples,
    )

    pngs = list_rendered_pngs(out_dir)
    if not pngs:
        print("ERROR: No rendered PNGs found.", file=sys.stderr)
        sys.exit(3)

    # 2) Gemini
    client = gemini_client(api_key=args.api_key)
    run_id = str(uuid.uuid4())
    results_path = args.results or os.path.join(out_dir, "gemini_results.jsonl")

    print(f"Sending {len(pngs)} images to Gemini in one request...")
    resp = send_all_images_to_gemini_one_prompt(
        client=client,
        model=args.model,
        prompt=GEMINI_PROMPT,
        image_paths=pngs,
    )

    raw = resp.model_dump(mode="json") if hasattr(resp, "model_dump") else None
    record = {
        "run_id": run_id,
        "glb_path": glb_abs,
        "render_dir": out_dir,
        "image_paths": [os.path.abspath(p) for p in pngs],
        "model": args.model,
        "prompt": GEMINI_PROMPT,
        "response_text": getattr(resp, "text", None),
        "raw": raw,
        "ts": time.time(),
    }
    append_record_jsonl(results_path, record)
    print(f"Gemini response appended to: {results_path}")
    print(f"run_id: {run_id}")

    # 3) Read correct output back
    record_from_file = find_record_by_run_id(results_path, run_id)
    target_count = extract_object_count_from_record(record_from_file)
    print(f"[gemini] Parsed target object_count = {target_count}")

    # 4) Find eps exact OR closest
    best_eps, best_clusters, exact, top10 = find_eps_exact_or_closest(
        input_glb=glb_abs,
        target_clusters=target_count,
        eps_step=args.eps_step,
        min_samples=args.min_samples,
        eps_min=args.eps_min,
        eps_max=args.eps_max,
    )

    if not exact:
        print("\n⚠️  Could not find an eps that matches Gemini's object count EXACTLY.")
        print(f"    Gemini target: {target_count}")
        print(f"    Closest found: {best_clusters} clusters at eps={best_eps:.3f}")
        print("    Top 10 closest candidates (eps -> clusters):")
        for e, c in top10:
            print(f"      {e:.3f} -> {c}")

    # 5) Split/export anyway using best_eps
    split_out = os.path.abspath(args.split_out)
    written = split_glb_with_eps(
        input_glb=glb_abs,
        out_dir=split_out,
        eps=best_eps,
        min_samples=args.min_samples,
    )

    print("\n✅ Done.")
    print(f"Gemini target count: {target_count}")
    print(f"Selected eps: {best_eps:.3f} (exact match: {exact})")
    print(f"Resulting clusters exported: {written}")
    print(f"Wrote GLB files to: {split_out}")
    print(f"Gemini JSONL: {results_path}")
    print(f"run_id: {run_id}")


if __name__ == "__main__":
    main()
