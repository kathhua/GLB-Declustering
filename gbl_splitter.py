#!/usr/bin/env python3
"""
Split an asset-pack GLB (e.g. 10 cars in a circle) into separate GLB files,
one per object.

Strategy
--------
1. Load the file as a trimesh.Scene.
2. For every node that has geometry:
    - Get its world transform.
    - Copy the mesh and apply the transform (so vertices are in world space).
    - Record the mesh's bounding-box center.
3. Run DBSCAN on the centers to group nearby meshes into clusters
   (each cluster ≈ one car).
4. For each cluster, concatenate meshes and export to its own .glb.

You may need to tweak --eps depending on the scale/spacing of your models.
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import trimesh
from sklearn.cluster import DBSCAN


def extract_world_meshes(scene: trimesh.Scene):
    """
    From a trimesh.Scene, return:

    meshes_world: [Trimesh, ...] each already transformed to world space
    centers:      (N, 3) array of bounding-box centers for each mesh
    """
    meshes_world = []
    centers = []

    # scene.graph.nodes_geometry is a list of node names that have geometry
    for node_name in scene.graph.nodes_geometry:
        # Get transform from world -> this node, and the geometry name
        matrix, geom_name = scene.graph.get(frame_to=node_name)

        base_mesh = scene.geometry[geom_name]
        mesh = base_mesh.copy()
        mesh.apply_transform(matrix)

        meshes_world.append(mesh)
        centers.append(mesh.bounding_box.centroid)

    centers = np.vstack(centers) if centers else np.zeros((0, 3))
    return meshes_world, centers


def cluster_meshes(centers: np.ndarray, eps: float, min_samples: int):
    """
    Cluster meshes based on their centers using DBSCAN.

    Returns:
        labels: array of cluster labels for each center (shape (N,))
                -1 means "noise" (not part of any cluster).
    """
    if len(centers) == 0:
        return np.array([], dtype=int)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(centers)
    return labels


def split_glb(
    input_glb: str,
    out_dir: str,
    eps: float = 3.0,
    min_samples: int = 1,
):
    # Load scene
    scene = trimesh.load(input_glb, force="scene")

    # Extract meshes in world space & their centers
    meshes_world, centers = extract_world_meshes(scene)

    if not meshes_world:
        raise RuntimeError("No geometry nodes found in the GLB.")

    # Cluster by spatial proximity
    labels = cluster_meshes(centers, eps=eps, min_samples=min_samples)

    # Group meshes by cluster label
    groups = defaultdict(list)
    for mesh, label in zip(meshes_world, labels):
        # If DBSCAN marks something as noise (-1), treat it as its own cluster
        if label == -1:
            label = max(labels) + 1  # put all noise into a new cluster
        groups[label].append(mesh)

    base_name = os.path.splitext(os.path.basename(input_glb))[0]
    os.makedirs(out_dir, exist_ok=True)

    print(f"Found {len(groups)} object clusters")

    for i, (label, mesh_list) in enumerate(sorted(groups.items()), start=1):
        # Combine all meshes in this cluster into one mesh
        if len(mesh_list) == 1:
            combined = mesh_list[0]
        else:
            combined = trimesh.util.concatenate(mesh_list)

        out_path = os.path.join(out_dir, f"{base_name}_object_{i:02d}.glb")
        combined.export(out_path)
        print(f"  -> wrote {out_path}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Split a GLB asset pack into individual GLB files."
    )
    parser.add_argument("input_glb", help="Path to input .glb file (e.g. cars.glb)")
    parser.add_argument(
        "-o",
        "--outdir",
        default="split_objects",
        help="Output directory for per-object GLBs",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=3.0,
        help=(
            "DBSCAN eps (distance threshold). "
            "Increase if objects are spread out; decrease if they merge."
        ),
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="DBSCAN min_samples; 1 works well for this use-case.",
    )

    args = parser.parse_args()
    split_glb(args.input_glb, args.outdir, eps=args.eps, min_samples=args.min_samples)


if __name__ == "__main__":
    main()



# to-do 
"  Make inference model to identify ideal eps. add a gemini checker"