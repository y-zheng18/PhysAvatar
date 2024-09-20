# Modification of the original code from https://github.com/rin-23/RobustSkinWeightsTransferCode
# MIT License
#
# Copyright (c) 2024 Rinat Abdrashitov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import numpy as np
from pytorch3d.ops import knn_points
import torch
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from scipy.spatial import cKDTree
import robust_laplacian

from utils.smplx_deformer import SmplxDeformer


def compute_vertex_normals(vertices, faces):
    """Compute vertex normals from faces."""
    # numpy version
    v_normals = np.zeros(vertices.shape, dtype=np.float32)
    for face in faces:
        v1 = vertices[face[0]]
        v2 = vertices[face[1]]
        v3 = vertices[face[2]]
        normal = np.cross(v2 - v1, v3 - v1)
        v_normals[face[0]] += normal
        v_normals[face[1]] += normal
        v_normals[face[2]] += normal
    v_normals = v_normals / np.linalg.norm(v_normals, axis=1, keepdims=True)
    return v_normals

def find_closest_points(v1, v2):
    """
    Find the closest points between two meshes v1 and v2
    :param v1: vertices of mesh 1
    :param v2: vertices of mesh 2
    :return: dist, idx
    """
    v1 = torch.from_numpy(v1)
    v2 = torch.from_numpy(v2)
    knn_output = knn_points(v1.unsqueeze(0), v2.unsqueeze(0), K=1)
    return knn_output.dists.squeeze().numpy(), knn_output.idx.squeeze().numpy()

"""
https://github.com/rin-23/RobustSkinWeightsTransferCode
"""
def filter_high_confidence_matches(target_vertex_data, closest_points_data, max_distance, max_angle):
    """filter high confidence matches using structured arrays."""
    print("Filtering high confidence matches...")
    target_positions = target_vertex_data["vertices"]
    target_normals = target_vertex_data["normal"]
    source_positions = closest_points_data["vertices"]
    source_normals = closest_points_data["normal"]
    print(source_positions.shape, target_positions.shape)

    # Calculate distances (vectorized)
    distances = np.linalg.norm(source_positions - target_positions, axis=1)

    # Calculate angles between normals (vectorized)
    cos_angles = np.einsum("ij,ij->i", source_normals, target_normals)
    cos_angles /= np.linalg.norm(source_normals, axis=1) * np.linalg.norm(target_normals, axis=1)
    # cos_angles = np.abs(cos_angles)  # Consider opposite normals by taking absolute value
    angles = np.arccos(np.clip(cos_angles, -1, 1)) * 180 / np.pi

    # Apply thresholds (vectorized)
    high_confidence_indices = np.where((distances <= max_distance) & (angles <= max_angle))[0]

    return high_confidence_indices.tolist()

def add_laplacian_entry_in_place(L, tri_positions, tri_indices):
    # type: (sp.lil_matrix, np.ndarray, np.ndarray) -> None
    """add laplacian entry.

    CAUTION: L is modified in-place.
    """

    i1 = tri_indices[0]
    i2 = tri_indices[1]
    i3 = tri_indices[2]

    v1 = tri_positions[0]
    v2 = tri_positions[1]
    v3 = tri_positions[2]

    # calculate cotangent
    cotan1 = compute_cotangent(v2, v1, v3)
    cotan2 = compute_cotangent(v1, v2, v3)

    # update laplacian matrix
    L[i1, i2] += cotan1  # type: ignore
    L[i2, i1] += cotan1  # type: ignore
    L[i1, i1] -= cotan1  # type: ignore
    L[i2, i2] -= cotan1  # type: ignore

    L[i2, i3] += cotan2  # type: ignore
    L[i3, i2] += cotan2  # type: ignore
    L[i2, i2] -= cotan2  # type: ignore
    L[i3, i3] -= cotan2  # type: ignore


def add_area_in_place(areas, tri_positions, tri_indices):
    # type: (np.ndarray, np.ndarray, np.ndarray) -> None
    """add area.

    CAUTION: areas is modified in-place.
    """

    v1 = tri_positions[0]
    v2 = tri_positions[1]
    v3 = tri_positions[2]
    area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

    for idx in tri_indices:
        areas[idx] += area


def compute_laplacian_and_mass_matrix(mesh_v, mesh_f):
    """compute laplacian matrix from mesh.

    treat area as mass matrix.
    """

    # initialize sparse laplacian matrix

    n_vertices = mesh_v.shape[0]
    L = sp.lil_matrix((n_vertices, n_vertices))
    areas = np.zeros(n_vertices)

    for i in range(mesh_f.shape[0]):

        tri_positions = mesh_v[mesh_f[i]]
        tri_indices = mesh_f[i]

        add_laplacian_entry_in_place(L, tri_positions, tri_indices)
        add_area_in_place(areas, tri_positions, tri_indices)

    L_csr = L.tocsr()
    M_csr = sp.diags(areas)

    return L_csr, M_csr


def compute_cotangent(v1, v2, v3):
    """compute cotangent from three points."""

    edeg1 = v2 - v1
    edeg2 = v3 - v1

    area = np.linalg.norm(np.cross(edeg1, edeg2))
    cotan = np.dot(edeg1, edeg2) / area

    return cotan


def compute_mass_matrix(mesh_v, mesh_f):
    """Compute the mass matrix for a given mesh.

    Args:
        mesh_v (np.ndarray): The vertices of the mesh.
        mesh_f (np.ndarray): The faces of the mesh.
    Returns:
        sp.dia_matrix: The diagonal sparse mass matrix, where each diagonal element represents
                       the total area associated with a vertex.
    """

    n_vertices = mesh_v.shape[0]
    areas = np.zeros(n_vertices)
    for i in range(mesh_f.shape[0]):
        tri_positions = mesh_v[mesh_f[i]]   # (3, 3)
        tri_indices = mesh_f[i]

        v1 = np.array(tri_positions[0])
        v2 = np.array(tri_positions[1])
        v3 = np.array(tri_positions[2])

        # calculate area of the current face
        area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

        # add area to the corresponding vertices
        for idx in tri_indices:
            areas[idx] += area

    # create sparse diagonal mass matrix
    M = sp.diags(areas)

    return M

def calculate_threshold_distance(mesh_v, threadhold_ratio=0.05):

    """Returns dbox * 0.05
    dbox is the target mesh bounding box diagonal length.
    """

    bbox_min = np.min(mesh_v, axis=0)
    bbox_max = np.max(mesh_v, axis=0)
    bbox_diag = bbox_max - bbox_min
    bbox_diag_length = np.linalg.norm(bbox_diag)

    threshold_distance = bbox_diag_length * threadhold_ratio

    return threshold_distance

def segregate_vertices_by_confidence(src_mesh, dst_mesh, threshold_distance=0.05, threshold_angle=15.0, use_kdtree=False):
    """segregate vertices by confidence."""

    threshold_distance = calculate_threshold_distance(dst_mesh['vertices'], threshold_distance)

    closest_dist, closest_idx = find_closest_points(dst_mesh['vertices'], src_mesh['vertices'])
    print(closest_dist.shape, closest_idx.shape)
    closest_points_data = {k: v[closest_idx] for k, v in src_mesh.items()}

    confident_vertex_indices = filter_high_confidence_matches(dst_mesh, closest_points_data, threshold_distance, threshold_angle)
    unconvinced_vertex_indices = list(set(range(dst_mesh['vertices'].shape[0])) - set(confident_vertex_indices))

    return confident_vertex_indices, unconvinced_vertex_indices


def __do_inpainting(mesh_v, mesh_f, known_weights):
    print('computing laplacian and mass matrix...')
    L, M = robust_laplacian.mesh_laplacian(mesh_v, mesh_f) #compute_laplacian_and_mass_matrix(mesh_v, mesh_f)
    print('computing laplacian and mass matrix done.')
    Q = -L + L @ sp.diags(np.reciprocal(M.diagonal())) @ L

    S_match = np.array(list(known_weights.keys()))
    S_nomatch = np.array(list(set(range(mesh_v.shape[0])) - set(S_match)))

    Q_UU = sp.csr_matrix(Q[np.ix_(S_nomatch, S_nomatch)])
    Q_UI = sp.csr_matrix(Q[np.ix_(S_nomatch, S_match)])

    num_vertices = mesh_v.shape[0]
    num_bones = len(next(iter(known_weights.values())))

    W = np.zeros((num_vertices, num_bones))
    for i, weights in known_weights.items():
        W[i] = weights

    W_I = W[S_match, :]
    W_U = W[S_nomatch, :]

    for bone_idx in range(num_bones):
        b = -Q_UI @ W_I[:, bone_idx]
        W_U[:, bone_idx] = splinalg.spsolve(Q_UU, b)

    W[S_nomatch, :] = W_U

    # apply constraints,

    # each element is between 0 and 1
    W = np.clip(W, 1e-10, 1.0)

    # normalize each row to sum to 1
    W = W / (W.sum(axis=1, keepdims=True) + 1e-10)

    return W


def calculate_inpainting(mesh_v, mesh_f, lbs_w, unknown_vertex_indices):
    """Inpainting weights for unknown vertices from known vertices."""

    num_vertices = mesh_v.shape[0]
    known_indices = list(set(range(num_vertices)) - set(unknown_vertex_indices))

    known_weights = {}  # type: Dict[int, np.ndarray]
    for vertex_index in known_indices:
        known_weights[vertex_index] = lbs_w[vertex_index]

    return __do_inpainting(mesh_v, mesh_f, known_weights)


def compute_weights_for_remaining_vertices(target_mesh, known_weights):
    """compute weights for remaining vertices."""

    try:
        optimized = __do_inpainting(target_mesh['vertices'], target_mesh['faces'], known_weights)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Error: {}".format(e))
        raise

    return optimized



def lbs_weights_inpainting(
        smplx_gender: str='neutral',
        src_mesh_path: str=None,
        src_param_path: str=None,
        target_mesh_path: str=None,
        output_path: str=None,
):
    # setup
    lbs_deformer = SmplxDeformer(gender=smplx_gender)
    src_v, src_f = lbs_deformer.read_obj(src_mesh_path)

    src_v_barycenter = src_v[src_f].mean(axis=1, keepdims=True)    # (N, 1, 3)
    v1 = src_v[src_f[:, 0]]
    v2 = src_v[src_f[:, 1]]
    v3 = src_v[src_f[:, 2]]
    src_vn = np.cross(v2 - v1, v3 - v1) # (N, 3)
    src_vn = src_vn / np.linalg.norm(src_vn, axis=1, keepdims=True)

    tar_v, tar_f = lbs_deformer.read_obj(target_mesh_path)
    tar_vn = compute_vertex_normals(tar_v, tar_f)
    smplx_param = torch.load(src_param_path,
                             map_location=torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda'))
    smplx = lbs_deformer.smplx_forward(smplx_param)
    smplx_vn = compute_vertex_normals(src_v, src_f)
    t_human_v, transform_matrix, lbs_w = lbs_deformer.transform_to_t_pose(torch.from_numpy(tar_v).unsqueeze(0),
                                                                          smplx,
                                                                          smplx_param['trans'],
                                                                          smplx_param['scale'],
                                                                          v_normals=torch.from_numpy(tar_vn).unsqueeze(0),
                                                                          smplx_normals=torch.from_numpy(smplx_vn).unsqueeze(0),
                                                                          k=10)
    lbs_w = lbs_w[0].detach().cpu().numpy()

    src_mesh = {'vertices': src_v_barycenter[:, 0], 'normal': src_vn}
    tgt_mesh = {'vertices': tar_v, 'faces': tar_f, 'normal': tar_vn}

    confident_vertex_indices, unconvinced_vertex_indices = segregate_vertices_by_confidence(src_mesh, tgt_mesh, threshold_distance=0.05, threshold_angle=15.0)
    print(len(confident_vertex_indices), len(unconvinced_vertex_indices))

    # save vertices
    vis_path = os.path.join(output_path, 'vis')
    os.makedirs(vis_path, exist_ok=True)
    mesh_path = os.path.join(vis_path, 'vis_correspondence_mesh.obj')
    with open(mesh_path, 'w') as f:
        for i, v in enumerate(tar_v):
            if i in confident_vertex_indices:
                f.write(f"v {v[0]} {v[1]} {v[2]} 0 0.8 0\n")
            elif i in unconvinced_vertex_indices:
                f.write(f"v {v[0]} {v[1]} {v[2]} 1 0 0\n")
            else:
                f.write(f"v {v[0]} {v[1]} {v[2]} 0 1 0\n")
        for face in tar_f:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


    closest_dist, closest_idx = find_closest_points(tgt_mesh['vertices'], src_mesh['vertices'])

    known_weights = {}

    for i, idx in enumerate(closest_idx):
        if i in confident_vertex_indices:
            known_weights[i] = lbs_w[i] #lbs_smplx[idx]
    print(len(known_weights.keys()))
    # inpainting
    optimized_weights = compute_weights_for_remaining_vertices(tgt_mesh, known_weights)
    np.save(os.path.join(output_path, 'optimized_weights.npy'), optimized_weights)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--smplx_gender', type=str, default='neutral')
    parser.add_argument('--src_mesh_path', type=str, default='./data/a1_s1/smplx_fitted/000460.obj')
    parser.add_argument('--src_param_path', type=str, default='./data/a1_s1/smplx_fitted/000460.pth')
    parser.add_argument('--target_mesh_path', type=str, default='./data/a1_s1/FrameRec000460.obj')
    parser.add_argument('--output_path', type=str, default='./data/a1_s1/')
    args = parser.parse_args()

    lbs_weights_inpainting(args.smplx_gender,
                           args.src_mesh_path,
                           args.src_param_path,
                           args.target_mesh_path,
                           args.output_path)
