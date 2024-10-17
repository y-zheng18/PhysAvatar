import argparse
import glob
import os
import numpy as np
import trimesh 
from mi_opt_util import interpolate_vertex_colors
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

def parse_args():
    parser = argparse.ArgumentParser(description="Export colored PLY file")
    parser.add_argument("--mesh_path", type=str, default="./FrameRec000460.obj", help="Path to initial obj file")
    parser.add_argument("--npz_path", type=str, default="../output/actor01/seq01/", help="Path to npz files")
    parser.add_argument("--name", type=str, default="a1/a1s1", help="Name for output files")
    parser.add_argument("--output_dir", type=str, default="./data/", help="Output directory")
    return parser.parse_args()

def write_obj(filename, vertices, faces):
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    return

def main(args):
    # Load initial obj file
    mesh = trimesh.load_mesh(args.mesh_path)
    v = mesh.vertices
    f = mesh.faces

    # Load all npz files
    paths = sorted(glob.glob(os.path.join(args.npz_path, "*.npz")))
    colors = []
    vertices = []
    print(paths[0])
    data_ = np.load(paths[0])
    colors.append(data_["rgb_colors"])
    vertices.append(data_["vertices"])
    face = data_["faces"]
    colors = np.array(colors)
    color = interpolate_vertex_colors(face, colors[0], len(vertices[0]))

    mesh = trimesh.Trimesh(vertices=vertices[0], faces=face)
    print(vertices[0].shape)
    write_obj(f"{args.output_dir}/{args.name}_export_mi.obj", vertices[0], face)
    face = face.reshape(-1)
    _, idx = np.unique(face, return_index=True)
    order = face[np.sort(idx)]
    print("before mitsuba", vertices[0].shape)
    v_old = vertices[0]

    mesh = mi.load_dict({
        "type": "obj",
        "filename": f"{args.output_dir}/{args.name}_export_mi.obj",
        "face_normals": True,
    })
    params = mi.traverse(mesh)
    vertices = np.array(params['vertex_positions']).reshape(-1,3)
    print("after mitsuba", vertices.shape)

    # Find matching vertices
    index_to_older = [np.where((v_old == v).all(axis=1))[0][0] for v in vertices]
    np.save(f"{args.output_dir}{args.name}index_to_older.npy", index_to_older)

    faces = np.array(params['faces']).reshape(-1,3)
    mesh_reordered = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh_reordered.visual.vertex_colors = color[order]
    mesh_reordered.export(f"{args.output_dir}/{args.name}_export_colored_mi_reorder.ply")

if __name__ == "__main__":
    args = parse_args()
    main(args)
