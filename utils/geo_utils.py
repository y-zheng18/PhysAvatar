import torch
from pytorch3d.transforms import matrix_to_quaternion

def normalize(x):
    norms = torch.norm(x, dim=1, keepdim=True)
    return x / norms

def compute_vertex_normals(vertices, faces):
    normals = torch.zeros_like(vertices)
    triangles = vertices[faces]

    e1 = triangles[:, 0] - triangles[:, 1]
    e2 = triangles[:, 2] - triangles[:, 1]
    n = torch.cross(e1, e2)

    for i in range(faces.shape[1]):
        normals.index_add_(0, faces[:, i], n)

    return normalize(normals)

def compute_face_normals(vertices, faces):
    # Compute the normal of each face
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    e1 = v2 - v1  # Edge 1
    e2 = v3 - v2  # Edge 2
    face_normals = torch.cross(e1, e2, dim=1)
    face_normals = face_normals / torch.norm(face_normals, dim=1, keepdim=True)
    return face_normals

def compute_face_barycenters(vertices, faces):
    # Compute the barycenter of each face
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    face_barycenters = (v1 + v2 + v3) / 3
    return face_barycenters

def compute_q_from_faces(vertices, faces, face_normals):
    # Edges
    v1 = vertices[faces[:, 0]]  # (N, 3)
    v2 = vertices[faces[:, 1]]  # (N, 3)
    v3 = vertices[faces[:, 2]]  # (N, 3)

    e1 = v2 - v1  # Edge 1
    e2 = v3 - v2  # Edge 2
    e3 = v1 - v3  # Edge 3

    # Lengths of edges
    len_e1 = e1.norm(dim=1)
    len_e2 = e2.norm(dim=1)
    len_e3 = e3.norm(dim=1)

    # Find the longest edge
    max_len, max_idx = torch.stack([len_e1, len_e2, len_e3], dim=1).max(dim=1, keepdim=True)

    # Use the longest edge as the x-axis
    x_axis = torch.where(max_idx == 0, e1, torch.where(max_idx == 1, e2, e3))
    x_axis = x_axis / x_axis.norm(dim=1, keepdim=True)  # Normalize

    # z-axis is the face normal
    z_axis = face_normals / face_normals.norm(dim=1, keepdim=True)  # Normalize

    # y-axis is the cross product of z and x
    y_axis = torch.cross(z_axis, x_axis, dim=1)
    y_axis = y_axis / y_axis.norm(dim=1, keepdim=True)  # Normalize

    # Construct the rotation matrix for each face
    rotation_matrices = torch.stack([x_axis, y_axis, z_axis], dim=2)

    # Convert rotation matrices to quaternions
    quaternions = matrix_to_quaternion(rotation_matrices)

    return quaternions

def compute_face_areas(vertices, faces):
    # Compute the area of each face
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    e1 = v2 - v1  # Edge 1
    e2 = v3 - v2  # Edge 2
    face_areas = torch.norm(torch.cross(e1, e2, dim=1), dim=1) / 2
    return face_areas