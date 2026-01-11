#!/usr/bin/env python3
"""
glb_render_gemini_then_split_vertex_dbscan.py

END-TO-END PIPELINE:
1) Render GLB -> PNGs with Blender (headless)
2) Send all PNGs to Gemini (one prompt) to get object count
3) Save JSONL record with run_id
4) Read the correct record back by run_id
5) Auto-search DBSCAN eps (vertex-distance DBSCAN) to match Gemini count exactly (else closest)
6) Export per-object GLBs using the selected eps

Key change vs center-based version:
- DBSCAN is run on a point cloud built from (sampled) WORLD-SPACE VERTICES.

Requirements:
  - Blender installed
  - pip install -U google-genai trimesh numpy scikit-learn

Auth:
  - export GEMINI_API_KEY="..."

Usage:
  python glb_render_gemini_then_split_vertex_dbscan.py input.glb --out gemini_runs --split-out split_objects

Performance knobs:
  --max-verts-per-mesh 2000   (default)
  --seed 0                   (default)
  --eps-step 0.05            (default)
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
# 4) Vertex-based DBSCAN splitting
# ============================================================
def extract_world_meshes(scene: trimesh.Scene) -> List[trimesh.Trimesh]:
    meshes_world: List[trimesh.Trimesh] = []
    for node_name in scene.graph.nodes_geometry:
        matrix, geom_name = scene.graph.get(frame_to=node_name)
        if geom_name is None:
            continue
        base_mesh = scene.geometry[geom_name]
        mesh = base_mesh.copy()
        mesh.apply_transform(matrix)
        meshes_world.append(mesh)
    return meshes_world


def sample_mesh_vertices(mesh: trimesh.Trimesh, max_verts: int, rng: np.random.Generator) -> np.ndarray:
    v = mesh.vertices
    if v is None or len(v) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    if len(v) <= max_verts:
        return v

    idx = rng.choice(len(v), size=max_verts, replace=False)
    return v[idx]


def build_vertex_cloud(
    meshes_world: List[trimesh.Trimesh],
    max_verts_per_mesh: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      points: (M,3) float32 array of sampled world-space vertices
      point_to_mesh: (M,) int array mapping each point to its mesh index
    """
    rng = np.random.default_rng(seed)
    pts_list: List[np.ndarray] = []
    map_list: List[np.ndarray] = []

    for mi, mesh in enumerate(meshes_world):
        pts = sample_mesh_vertices(mesh, max_verts_per_mesh, rng)
        if len(pts) == 0:
            continue
        pts_list.append(pts)
        map_list.append(np.full((len(pts),), mi, dtype=np.int32))

    if not pts_list:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    points = np.vstack(pts_list).astype(np.float32)
    point_to_mesh = np.concatenate(map_list).astype(np.int32)
    return points, point_to_mesh


def mesh_labels_from_point_labels(
    point_labels: np.ndarray,
    point_to_mesh: np.ndarray,
    num_meshes: int,
) -> np.ndarray:
    """
    Assign each mesh a label by majority vote of its sampled points.
    Noise points (-1) do not vote.
    If a mesh has only noise points, it stays -1.
    """
    mesh_labels = np.full((num_meshes,), -1, dtype=np.int32)

    for mi in range(num_meshes):
        mask = (point_to_mesh == mi)
        labs = point_labels[mask]
        labs = labs[labs != -1]
        if labs.size == 0:
            continue
        # majority vote
        vals, counts = np.unique(labs, return_counts=True)
        mesh_labels[mi] = int(vals[np.argmax(counts)])

    return mesh_labels


def relabel_noise_meshes_as_unique(mesh_labels: np.ndarray) -> np.ndarray:
    """
    For any mesh still labeled -1, assign it a fresh unique label so it becomes its own cluster.
    """
    out = mesh_labels.copy()
    next_label = int(out.max()) + 1 if out.size else 0
    for i in range(len(out)):
        if out[i] == -1:
            out[i] = next_label
            next_label += 1
    return out


def count_clusters(mesh_labels: np.ndarray) -> int:
    if mesh_labels.size == 0:
        return 0
    return len(set(int(x) for x in mesh_labels))


def split_glb_with_vertex_dbscan(
    input_glb: str,
    out_dir: str,
    eps: float,
    min_samples: int,
    max_verts_per_mesh: int,
    seed: int,
) -> int:
    scene = trimesh.load(input_glb, force="scene")
    meshes_world = extract_world_meshes(scene)
    if not meshes_world:
        raise RuntimeError("No geometry nodes found in the GLB.")

    points, point_to_mesh = build_vertex_cloud(meshes_world, max_verts_per_mesh, seed=seed)
    if len(points) == 0:
        raise RuntimeError("No vertices found to cluster (vertex cloud empty).")

    labels_points = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)

    mesh_labs = mesh_labels_from_point_labels(labels_points, point_to_mesh, num_meshes=len(meshes_world))
    mesh_labs = relabel_noise_meshes_as_unique(mesh_labs)

    groups: Dict[int, List[trimesh.Trimesh]] = defaultdict(list)
    for mi, lab in enumerate(mesh_labs):
        groups[int(lab)].append(meshes_world[mi])

    base_name = os.path.splitext(os.path.basename(input_glb))[0]
    os.makedirs(out_dir, exist_ok=True)

    print(f"[split-vertex] eps={eps:.3f} -> clusters={len(groups)} (exporting...)")
    for i, (_, mesh_list) in enumerate(sorted(groups.items()), start=1):
        combined = mesh_list[0] if len(mesh_list) == 1 else trimesh.util.concatenate(mesh_list)
        out_path = os.path.join(out_dir, f"{base_name}_object_{i:02d}.glb")
        combined.export(out_path)
        print(f"  -> wrote {out_path}")

    return len(groups)


def find_eps_exact_or_closest_vertex_dbscan(
    input_glb: str,
    target_clusters: int,
    eps_step: float,
    min_samples: int,
    max_verts_per_mesh: int,
    seed: int,
    eps_min: Optional[float] = None,
    eps_max: Optional[float] = None,
) -> Tuple[float, int, bool, List[Tuple[float, int]]]:
    """
    Sweeps eps for vertex-DBSCAN clustering, returns exact match if possible else closest.
    """
    scene = trimesh.load(input_glb, force="scene")
    meshes_world = extract_world_meshes(scene)
    if not meshes_world:
        raise RuntimeError("No geometry nodes found in the GLB.")

    points, point_to_mesh = build_vertex_cloud(meshes_world, max_verts_per_mesh, seed=seed)
    if len(points) == 0:
        raise RuntimeError("No vertices found to cluster (vertex cloud empty).")

    # Determine eps range from point cloud scale
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    diag = float(np.linalg.norm(pmax - pmin))
    diag = max(diag, 1e-6)

    if eps_min is None:
        eps_min = max(0.01, diag * 0.002)
    if eps_max is None:
        eps_max = diag * 0.25  # vertex DBSCAN needs much smaller eps than center clustering

    max_iters = int(np.ceil((eps_max - eps_min) / eps_step)) + 2

    tried: List[Tuple[float, int]] = []
    best_eps = float(eps_min)
    best_clusters = -1
    best_diff = 10**18
    exact = False

    print(f"[auto-eps-vertex] Target={target_clusters}")
    print(f"[auto-eps-vertex] Searching eps in [{eps_min:.4f}, {eps_max:.4f}] step={eps_step:.4f} (max {max_iters} iters)")
    print(f"[auto-eps-vertex] Point cloud size: {len(points):,} points (max_verts_per_mesh={max_verts_per_mesh})")

    eps = float(eps_min)
    for _ in range(max_iters):
        if eps > eps_max + 1e-12:
            break

        labels_points = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
        mesh_labs = mesh_labels_from_point_labels(labels_points, point_to_mesh, num_meshes=len(meshes_world))
        mesh_labs = relabel_noise_meshes_as_unique(mesh_labs)
        n = count_clusters(mesh_labs)

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
            print(f"[auto-eps-vertex] Exact match found: eps={eps:.4f} -> clusters={n}")
            break

        eps = round(eps + eps_step, 12)

    top10 = sorted(tried, key=lambda x: (abs(x[1] - target_clusters), x[0]))[:10]
    return best_eps, best_clusters, exact, top10


# ============================================================
# 5) Main pipeline
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Render GLB -> Gemini count -> vertex-DBSCAN eps search -> split into per-object GLBs")

    ap.add_argument("glb", help="Path to input .glb asset pack")

    # Rendering
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

    # Gemini
    ap.add_argument("--model", default="gemini-3-flash-preview", help="Gemini model name")
    ap.add_argument("--api-key", default=None, help="Optional API key override (else GEMINI_API_KEY env var)")
    ap.add_argument("--results", default=None, help="Results JSONL (default: <out>/gemini_results.jsonl)")

    # Splitting output
    ap.add_argument("--split-out", default="split_objects", help="Output dir for per-object GLBs")

    # Vertex DBSCAN knobs
    ap.add_argument("--eps-step", type=float, default=0.01, help="eps sweep step size (vertex DBSCAN usually needs smaller steps)")
    ap.add_argument("--min-samples", type=int, default=8, help="DBSCAN min_samples for point cloud (default 8)")
    ap.add_argument("--eps-min", type=float, default=None, help="Optional eps min override")
    ap.add_argument("--eps-max", type=float, default=None, help="Optional eps max override")

    ap.add_argument("--max-verts-per-mesh", type=int, default=2000,
                    help="Sample up to this many vertices per mesh for DBSCAN (default 2000)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for vertex sampling (default 0)")

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

    # 4) Find eps exact OR closest using vertex DBSCAN
    best_eps, best_clusters, exact, top10 = find_eps_exact_or_closest_vertex_dbscan(
        input_glb=glb_abs,
        target_clusters=target_count,
        eps_step=args.eps_step,
        min_samples=args.min_samples,
        max_verts_per_mesh=args.max_verts_per_mesh,
        seed=args.seed,
        eps_min=args.eps_min,
        eps_max=args.eps_max,
    )

    if not exact:
        print("\n⚠️  Could not find an eps that matches Gemini's object count EXACTLY (vertex DBSCAN).")
        print(f"    Gemini target: {target_count}")
        print(f"    Closest found: {best_clusters} clusters at eps={best_eps:.4f}")
        print("    Top 10 closest candidates (eps -> clusters):")
        for e, c in top10:
            print(f"      {e:.4f} -> {c}")

    # 5) Split/export anyway using best_eps
    split_out = os.path.abspath(args.split_out)
    written = split_glb_with_vertex_dbscan(
        input_glb=glb_abs,
        out_dir=split_out,
        eps=best_eps,
        min_samples=args.min_samples,
        max_verts_per_mesh=args.max_verts_per_mesh,
        seed=args.seed,
    )

    print("\n✅ Done.")
    print(f"Gemini target count: {target_count}")
    print(f"Selected eps: {best_eps:.4f} (exact match: {exact})")
    print(f"Resulting clusters exported: {written}")
    print(f"Wrote GLB files to: {split_out}")
    print(f"Gemini JSONL: {results_path}")
    print(f"run_id: {run_id}")


if __name__ == "__main__":
    main()
