import torch
import torch.nn as nn

class FaceNormals(nn.Module):
    def __init__(self, normalize=True):
        super(FaceNormals, self).__init__()
        self.normalize = normalize

    def forward(self, vertices, faces):
        v = vertices
        f = faces

        v0 = v[f[..., 0]]
        v1 = v[f[..., 1]]
        v2 = v[f[..., 2]]
        e1 = v0 - v1
        e2 = v2 - v1
        face_normals = torch.cross(e2, e1, dim=-1)

        if self.normalize:
            face_normals = nn.functional.normalize(face_normals, dim=-1)

        return face_normals

class PairwiseDistance(nn.Module):
    def __init__(self):
        super(PairwiseDistance, self).__init__()

    def forward(self, A, B):
        rA = torch.sum(A ** 2, dim=-1)
        rB = torch.sum(B ** 2, dim=-1)
        distances = -2 * torch.matmul(A, B.transpose(-2, -1)) + rA.unsqueeze(-1) + rB.unsqueeze(-2)
        return distances

class NearestNeighbour(nn.Module):
    def __init__(self):
        super(NearestNeighbour, self).__init__()

    def forward(self, A, B):
        distances = PairwiseDistance()(A, B)
        nearest_neighbour = torch.argmin(distances, dim=-1)
        return nearest_neighbour.type(torch.int32)

