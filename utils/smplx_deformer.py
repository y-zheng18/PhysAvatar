import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
import smplx
import smplx.lbs as lbs

import pytorch3d.transforms

from human_body_prior.train.vposer_smpl import VPoser

from pytorch3d.ops import knn_points


class SmplxDeformer():
    def __init__(self, gender='neutral', num_betas=300):
        # self.renderer = pyrender.OffscreenRenderer(viewport_width=img_size[0], viewport_height=img_size[1])
        self.gender = gender
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.smplx_model = smplx.SMPLX(model_path='./data/body_models/smplx', ext='npz', gender=gender,
                                      num_betas=num_betas, num_expression_coeffs=100, use_face_contour=False,
                                      use_pca=False).eval().to(self.device)
        self.vposer = VPoser(512, 32, [3, 21]).eval().to(self.device)

        # download vposer at https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=vposer_v1_0.zip
        self.vposer.load_state_dict(torch.load('./data/body_models/TR00_E096.pt', map_location='cpu'))

    def read_obj(self, filename):
        # Parse the .obj file and return vertices and indices
        # This is a placeholder function, you need to implement the parsing
        # according to the .obj file format.
        vertices = []
        indices = []
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('v '):  # This line describes a vertex
                    parts = line.strip().split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):  # This line describes a face
                    parts = line.strip().split()
                    face_indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]  # OBJ indices start at 1
                    indices.append(face_indices)
            vertices = np.array(vertices, dtype=np.float32)
            indices = np.array(indices, dtype=np.int32)

        return vertices, indices

    def save_obj(self, filename, v, f):
        with open(filename, 'w') as fp:
            for vi in v:
                fp.write('v %f %f %f\n' % (vi[0], vi[1], vi[2]))
            for fi in f:
                fp.write('f %d %d %d\n' % (fi[0] + 1, fi[1] + 1, fi[2] + 1))
        fp.close()

    def smplx_forward(self, smplx_param):
        if 'latent' in smplx_param:
            body_pose = pytorch3d.transforms.rotation_conversions.matrix_to_axis_angle(
                self.vposer.decode(smplx_param['latent']).view(-1, 3, 3)).view(1, -1)
        else:
            body_pose = smplx_param['body_pose']
        smplx_out = self.smplx_model.forward(
            transl=smplx_param['trans'],
            global_orient=smplx_param['orient'],
            body_pose=body_pose,
            betas=smplx_param['beta'],
            left_hand_pose=smplx_param['left_hand_pose'],
            right_hand_pose=smplx_param['right_hand_pose'],
            expression=smplx_param['expr'],
            jaw_pose=smplx_param['jaw_pose'],
            leye_pose=smplx_param['left_eye_pose'],
            reye_pose=smplx_param['right_eye_pose'],
            return_full_pose=True
        )

        full_pose = smplx_out.full_pose
        batch_size = body_pose.shape[0]
        # Concatenate the shape and expression coefficients
        scale = int(batch_size / smplx_param['beta'].shape[0])
        betas = smplx_param['beta']
        expression = smplx_param['expr']
        if scale > 1:
            betas = betas.expand(scale, -1)
            expression = expression.expand(scale, -1)
        shape_components = torch.cat([betas, expression], dim=-1)

        shapedirs = torch.cat([self.smplx_model.shapedirs, self.smplx_model.expr_dirs], dim=-1)

        # get transform matrix
        batch_size = max(shape_components.shape[0], full_pose.shape[0])
        device, dtype = shape_components.device, shape_components.dtype

        # Add shape contribution
        v_shaped = self.smplx_model.v_template + lbs.blend_shapes(shape_components, shapedirs)

        # Get the joints
        # NxJx3 array
        J = lbs.vertices2joints(self.smplx_model.J_regressor, v_shaped)

        # Add pose blend shapes
        # N x J x 3 x 3
        rot_mats = lbs.batch_rodrigues(full_pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        # Get the global joint location
        J_transformed, transform_mat = lbs.batch_rigid_transform(rot_mats, J, self.smplx_model.parents, dtype=dtype)

        scale = smplx_param['scale']
        if len(scale.shape):
            scale = scale.unsqueeze(dim=1)
        smplx_out.joints *= scale
        smplx_out.vertices *= scale
        smplx_out.transform_mat = transform_mat
        # smplx_out.transform_mat[:, :, :3, 3] /= scale
        # print(smplx_out.transform_mat.shape)
        return smplx_out

    def export(self, smplx_param, save_path):
        smplx_out = self.smplx_forward(smplx_param)
        with open(save_path, 'w') as smplx_save:
            for v in smplx_out.vertices[0].detach().cpu().numpy():
                smplx_save.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
            for f in self.smplx_model.faces:
                smplx_save.write('f {} {} {}\n'.format(f[0] + 1, f[1] + 1, f[2] + 1))

    def find_k_closest_verts(self, points, verts, k, points_normals=None, verts_normals=None, normal_weight=0.1):
        """

        Args:
            points: (bs, n, 3)
            verts: (bs, m, 3)
        Returns:
            dist: (bs, n, k) FloatTensor of squared distances
            idxs: (bs, n, k) LongTensor of vertex indices
        """
        assert points.device == verts.device
        assert points.shape[0] == 1 or verts.shape[0] == 1 or points.shape[0] == verts.shape[0], "batch size mismatch!"
        if points_normals is None:
            points_normals = torch.zeros_like(points)
        if verts_normals is None:
            verts_normals = torch.zeros_like(verts)
        assert points_normals.shape == points.shape
        assert verts_normals.shape == verts.shape
        points = torch.cat([points, normal_weight * points_normals], dim=-1)
        verts = torch.cat([verts, normal_weight * verts_normals], dim=-1)
        knn_output = knn_points(points, verts, K=k)
        return knn_output.dists, knn_output.idx

    def weights_from_k_closest_verts(self, points, verts, k, points_normals=None, verts_normals=None, normal_weight=0.1, p=1):
        """Compute weights from k closest vertices using "Shepard's Method".
        Args:
            points: (bs, n, 3)
            verts: (bs, m, 3)
            k: int
            points_normals: (bs, n, 3)
            verts_normals: (bs, m, 3)
            normal_weight: scalar to include normal
            p: int
        Returns:
            weights: (bs, n, k) FloatTensor of weights
            point_verts_idx: (bs, n, k) LongTensor of vertex indices
        """
        dists, point_verts_idx = self.find_k_closest_verts(points, verts, k, points_normals, verts_normals, normal_weight)
        dists = torch.clamp(dists, min=1e-8)
        weights = torch.pow(dists, -p)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights, point_verts_idx

    def transform_to_t_pose(self,
                            vertices,
                            smplx_output,
                            global_transl=None,
                            scale=None,
                            lbs_w=None,
                            k=10,
                            v_normals=None,
                            smplx_normals=None,
                            normal_weight=0.1):
        """
        Transform the given mesh to the T-pose using the inverse transformations of the SMPLX model.

        Args:
            vertices (torch.Tensor): The vertices of the mesh to be transformed.
            smplx_output (smplx.SMPLXOutput): The output of the SMPLX forward pass.
            weights (torch.Tensor): The blend skinning weights for each vertex.
            point_idx (torch.Tensor): Indices of the closest points in the SMPLX model.
        Returns:
            torch.Tensor: The transformed vertices in the T-pose.
        """

        transformation_matrix = smplx_output.transform_mat.to(self.device)
        lbs_weights_packed = self.smplx_model.lbs_weights  # (V, J+1)
        point_v_weights, point_v_idxs = self.weights_from_k_closest_verts(vertices, smplx_output.vertices, k=k, p=2,
                                                                          points_normals=v_normals, verts_normals=smplx_normals,
                                                                          normal_weight=normal_weight)
        if lbs_w is None:
            lbs_weights_pnts = torch.stack(
                [lbs_weights_packed[idxs] for idxs in point_v_idxs]
            )  # (bs, P, K, J+1)

            lbs_weights_pnts = torch.einsum(
                "npkj,npk->npj", lbs_weights_pnts, point_v_weights
            )
            B, num_points = point_v_weights.shape[:2]
            assert lbs_weights_pnts.shape == (
                B,
                num_points,
                self.smplx_model.lbs_weights.shape[-1],
            )
            # perform lbs
            # (N x V x (J + 1)) x (N x (J + 1) x 16)
            num_joints = self.smplx_model.J_regressor.shape[0]
            W = lbs_weights_pnts
            print(W.shape, W.sum(dim=-1))
        else:
            W = lbs_w #/ lbs_w.sum(dim=-1, keepdim=True).detach()
            transformation_matrix = smplx_output.transform_mat.to(self.device)
            num_joints = self.smplx_model.J_regressor.shape[0]
            B, num_points, _ = lbs_w.shape

        pose_offsets_smplx = smplx_output.vertices - smplx_output.v_shaped
        pose_offsets_pnts = pose_offsets_smplx[torch.arange(B).unsqueeze(1).unsqueeze(2), point_v_idxs]  # (B, P, K, 3)
        pose_offsets_pnts = torch.einsum('bpk,bpkj->bpj', point_v_weights, pose_offsets_pnts)  # Shape: (B, P, 3)

        T = torch.matmul(W, transformation_matrix.reshape((-1, B, num_joints, 16))).reshape((-1, B, num_points, 4, 4))
        T = torch.inverse(T)

        if len(scale.shape):
            scale = scale.unsqueeze(dim=1)
        vertices = vertices / scale
        if len(vertices.shape) == 3:
            global_transl = global_transl.unsqueeze(dim=1)
        vertices = vertices - global_transl
        src_verts = vertices.reshape((-1, num_points, 3)) #self.smplx_model.v_template.unsqueeze(dim=0).repeat(B, 1, 1)

        num_verts = src_verts.shape[-2]

        homogen_coord = torch.ones_like(
            src_verts.reshape((-1, B, num_verts, 3))[..., :1],
            dtype=src_verts.dtype,
            device=src_verts.device,
        )

        v_posed_homo = torch.cat([src_verts.reshape((-1, B, num_verts, 3)), homogen_coord], dim=-1)

        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))


        deformed_pnts = v_homo[..., :3, 0]

        return deformed_pnts, T, W

    def transform_to_pose(self, vertices, lbs_weights_pnts, smplx_output,  global_transl=None, scale=None):
        """
        Transform the given mesh to the T-pose using the inverse transformations of the SMPLX model.

        Args:
            vertices (torch.Tensor): The vertices of the mesh to be transformed.
            smplx_output (smplx.SMPLXOutput): The output of the SMPLX forward pass.
            weights (torch.Tensor): The blend skinning weights for each vertex.
            point_idx (torch.Tensor): Indices of the closest points in the SMPLX model.
        Returns:
            torch.Tensor: The transformed vertices in the T-pose.
        """

        transformation_matrix = smplx_output.transform_mat.to(self.device)
        B, num_points, _ = lbs_weights_pnts.shape
        # perform lbs
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = self.smplx_model.J_regressor.shape[0]
        W = lbs_weights_pnts #/ lbs_weights_pnts.sum(dim=-1, keepdim=True).detach()

        T = torch.matmul(W, transformation_matrix.reshape((-1, B, num_joints, 16))).reshape((-1, B, num_points, 4, 4))

        src_verts = vertices.reshape((-1, num_points, 3)) #self.smplx_model.v_template.unsqueeze(dim=0).repeat(B, 1, 1)

        num_verts = src_verts.shape[-2]

        homogen_coord = torch.ones_like(
            src_verts.reshape((-1, B, num_verts, 3))[..., :1],
            dtype=src_verts.dtype,
            device=src_verts.device,
        )

        v_posed_homo = torch.cat([src_verts.reshape((-1, B, num_verts, 3)), homogen_coord], dim=-1)

        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))


        deformed_pnts = v_homo[..., :3, 0]

        if global_transl is not None:
            deformed_pnts += global_transl.to(deformed_pnts.device).unsqueeze(dim=1)

        if scale is not None:
            if len(scale.shape):
                scale = scale.unsqueeze(dim=1)
            deformed_pnts *= scale

        return deformed_pnts, T


    def transform_to_t_pose_smplx(self, vertices, smplx_output, global_transl=None, scale=None, point_v_weights=None, point_v_idxs=None):
        """
        Transform the given mesh to the T-pose using the inverse transformations of the SMPLX model.

        Args:
            vertices (torch.Tensor): The vertices of the mesh to be transformed.
            smplx_output (smplx.SMPLXOutput): The output of the SMPLX forward pass.
            weights (torch.Tensor): The blend skinning weights for each vertex.
            point_idx (torch.Tensor): Indices of the closest points in the SMPLX model.
        Returns:
            torch.Tensor: The transformed vertices in the T-pose.
        """
        if point_v_weights is None:
            point_v_weights, point_v_idxs = self.weights_from_k_closest_verts(vertices, smplx_output.vertices, k=50, p=2)
            # print('point_v_weights', point_v_weights.shape, 'point_v_idxs', point_v_idxs.shape)
        # normalize weights

        transformation_matrix = smplx_output.transform_mat.to(self.device)

        point_v_weights = point_v_weights / point_v_weights.sum(dim=-1, keepdim=True)
        lbs_weights_packed = self.smplx_model.lbs_weights  # (V, J+1)

        lbs_weights_pnts = torch.stack(
            [lbs_weights_packed[idxs] for idxs in point_v_idxs]
        )  # (bs, P, K, J+1)

        lbs_weights_pnts = torch.einsum(
            "npkj,npk->npj", lbs_weights_pnts, point_v_weights
        )
        B, num_points = point_v_weights.shape[:2]
        assert lbs_weights_pnts.shape == (
            B,
            num_points,
            self.smplx_model.lbs_weights.shape[-1],
        )
        # perform lbs
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = self.smplx_model.J_regressor.shape[0]
        W = lbs_weights_pnts



        T = torch.matmul(W, transformation_matrix.reshape((-1, B, num_joints, 16))).reshape((-1, B, num_points, 4, 4))
        T = torch.inverse(T)

        if len(scale.shape):
            scale = scale.unsqueeze(dim=1)
        vertices = vertices / scale
        if len(vertices.shape) == 3:
            global_transl = global_transl.unsqueeze(dim=1)
        vertices = vertices - global_transl
        src_verts = vertices.reshape((-1, num_points, 3)) #self.smplx_model.v_template.unsqueeze(dim=0).repeat(B, 1, 1)

        num_verts = src_verts.shape[-2]

        homogen_coord = torch.ones_like(
            src_verts.reshape((-1, B, num_verts, 3))[..., :1],
            dtype=src_verts.dtype,
            device=src_verts.device,
        )

        v_posed_homo = torch.cat([src_verts.reshape((-1, B, num_verts, 3)), homogen_coord], dim=-1)

        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))


        deformed_pnts = v_homo[..., :3, 0]

        return deformed_pnts, T, W, point_v_weights, point_v_idxs

    def transform_to_pose_smplx(self, vertices, lbs_weights_pnts, smplx_output,  global_transl=None, scale=None):
        """
        Transform the given mesh to the T-pose using the inverse transformations of the SMPLX model.

        Args:
            vertices (torch.Tensor): The vertices of the mesh to be transformed.
            smplx_output (smplx.SMPLXOutput): The output of the SMPLX forward pass.
            weights (torch.Tensor): The blend skinning weights for each vertex.
            point_idx (torch.Tensor): Indices of the closest points in the SMPLX model.
        Returns:
            torch.Tensor: The transformed vertices in the T-pose.
        """

        transformation_matrix = smplx_output.transform_mat.to(self.device)
        B, num_points, _ = lbs_weights_pnts.shape
        # perform lbs
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = self.smplx_model.J_regressor.shape[0]
        W = lbs_weights_pnts

        T = torch.matmul(W, transformation_matrix.reshape((-1, B, num_joints, 16))).reshape((-1, B, num_points, 4, 4))

        src_verts = vertices.reshape((-1, num_points, 3)) #self.smplx_model.v_template.unsqueeze(dim=0).repeat(B, 1, 1)

        num_verts = src_verts.shape[-2]

        homogen_coord = torch.ones_like(
            src_verts.reshape((-1, B, num_verts, 3))[..., :1],
            dtype=src_verts.dtype,
            device=src_verts.device,
        )

        v_posed_homo = torch.cat([src_verts.reshape((-1, B, num_verts, 3)), homogen_coord], dim=-1)

        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))


        deformed_pnts = v_homo[..., :3, 0]

        if global_transl is not None:
            deformed_pnts += global_transl.to(deformed_pnts.device).unsqueeze(dim=1)

        if scale is not None:
            if len(scale.shape):
                scale = scale.unsqueeze(dim=1)
            deformed_pnts *= scale

        return deformed_pnts, T

