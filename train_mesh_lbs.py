import torch
import os
import json
import copy
import numpy as np
from PIL import Image
from random import randint
from tqdm import tqdm
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, o3d_knn, params2cpu

from external import calc_ssim, calc_psnr, update_params_and_optimizer, find_adjacent_faces
import trimesh
import wandb
import cv2
import glob
from utils.smplx_deformer import SmplxDeformer
from utils.geo_utils import compute_vertex_normals, compute_face_normals, \
    compute_face_barycenters, compute_q_from_faces, compute_face_areas
from losses.physics import collision_penalty


def get_dataset(t, md, seq,  data_path, downsample=False, downsample_view=1):
    dataset = []
    smplx_param_t = torch.load(f'./data/{seq}/smplx_fitted/{str(t).zfill(6)}.pth')
    smplx_param_t_1 = torch.load(f'./data/{seq}/smplx_fitted/{str(t + 1).zfill(6)}.pth')
    smplx_v, smplx_f, smplx_vn = read_obj(f'./data/{args.seq}/smplx_fitted/{str(t).zfill(6)}.obj')
    smplx_v = torch.from_numpy(smplx_v).cuda().float()
    smplx_f = torch.from_numpy(smplx_f).cuda().long()
    smplx_vn = compute_vertex_normals(smplx_v, smplx_f).cuda().float()

    if downsample_view == 8:
        selected_cam = ['Cam005', 'Cam008', 'Cam023', 'Cam037', 'Cam053', 'Cam055', 'Cam078', 'Cam079', 'Cam094', 'Cam096',
                        'Cam109', 'Cam112', 'Cam119', 'Cam127', 'Cam128', 'Cam131', 'Cam137', 'Cam133', 'Cam141', 'Cam121']
    else:
        selected_cam = list(md.keys())[::downsample_view]
    for i, c in enumerate(md.keys()):
        if c not in selected_cam:
            continue
        cam_info = md[c]
        w, h, k, c2w = cam_info['W'], cam_info['H'], cam_info['K'], cam_info['RT']
        w2c = np.linalg.inv(np.array(c2w))
        if downsample:
            w, h = w // 2, h // 2
            k = k / 2
            k[2, 2] = 1
        cam = setup_camera(w, h, k, w2c, near=1, far=10)

        cam_name = c.split('Cam')[1]
        im_name = str(t).zfill(6) + ".jpg"

        im = np.array(copy.deepcopy(Image.open(os.path.join(data_path, f"4x/rgbs/Cam{cam_name}/Cam{cam_name}_rgb{im_name}")))).astype(np.float32)
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        if downsample:
            im = torch.nn.functional.interpolate(im[None], scale_factor=0.5, mode='bilinear', align_corners=False)[0]
        dataset.append({'cam': cam, 'im': im, 'id': int(cam_name) - 1, 't': t,
                        'smplx_param': smplx_param_t, 'smplx_param_1': smplx_param_t_1,
                        'smplx_v': smplx_v, 'smplx_vn': smplx_vn})
    return dataset


def get_batch(todo_dataset, dataset, test_idx=[6, 126]):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    todo_dataset = [data for i, data in enumerate(todo_dataset) if i not in test_idx]
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data

def read_obj(filename):
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
        # Compute face normals
        face_normals = []
        for face in indices:
            v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            normal = normal / np.linalg.norm(normal)  # Normalize the vector
            face_normals.append(normal)

        face_normals = np.array(face_normals, dtype=np.float32)

    return vertices, indices, face_normals

def initialize_params(seq, md, obj_fn, cloth_fn):
    obj_fn = f"./data/{seq}/{obj_fn}"
    obj, faces, normals = read_obj(obj_fn)

    cloth_fn = f"./data/{seq}/{cloth_fn}"
    if not os.path.exists(cloth_fn):
        print(f"cloth_fn {cloth_fn} does not exist")
        cloth_fn = obj_fn

    max_cams = 160
    init_v = obj[:, :3]
    init_f = faces
    # compute barycentric coordinates of faces
    init_bary = compute_face_barycenters(init_v, init_f)
    sq_dist, _ = o3d_knn(init_bary, 4)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)

    init_means3D = init_bary
    unnorm_rotations = np.zeros((init_f.shape[0], 4))
    init_colors = np.zeros_like(init_f)
    init_scales = np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3))
    # only use two dimensions for scales, make sure that the third dimension is very small
    init_scales[:, 2] = -100
    init_log_opacities = np.zeros((init_f.shape[0], 1))

    params = {
        'vertices': init_v,
        'rgb_colors': init_colors,
        'logit_opacities': init_log_opacities,
        'log_scales': init_scales,
        'cam_m': np.zeros((max_cams, 3)),
        'cam_c': np.zeros((max_cams, 3)),
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    w2c = [np.linalg.inv(np.array(md[k]['RT'])) for k in md.keys()]


    cam_centers = np.linalg.inv(w2c)[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    face_neighbors = find_adjacent_faces(init_f, 3)

    means3D_neighbors = init_means3D[face_neighbors]

    neighbor_sq_dist = means3D_neighbors - init_means3D[:, None]
    neighbor_sq_dist = (neighbor_sq_dist ** 2).sum(-1)  # N, k=3

    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)

    variables = {'faces': torch.tensor(init_f).cuda().long(),
                 'means3D': torch.tensor(init_means3D).cuda().float(),
                 'unnorm_rotations': torch.tensor(unnorm_rotations).cuda().float(),
                 'max_2D_radius': torch.zeros(init_means3D.shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(init_means3D.shape[0]).cuda().float(),
                 'denom': torch.zeros(init_means3D.shape[0]).cuda().float(),
                 'face_neighbors': torch.tensor(face_neighbors).cuda().long(),
                 'neighbor_weight': torch.tensor(neighbor_weight).cuda().float().contiguous(),
                 'neighbor_dist': torch.tensor(neighbor_dist).cuda().float().contiguous(),}
    if os.path.exists(cloth_fn):
        cloth_obj, cloth_faces, cloth_normals = read_obj(cloth_fn)
        cloth_vertices = np.unique(cloth_faces.reshape(-1))
        cloth_vertices = torch.tensor(cloth_vertices).cuda().long()
        variables['cloth_v_idx'] = cloth_vertices
        print('cloth vertices', cloth_vertices.shape)

    return params, variables


def initialize_optimizer(params, variables, args):
    lrs = {
        'vertices': args.lr_means3D * variables['scene_radius'],
        'rgb_colors': args.lr_colors,
        'logit_opacities': 0.05,
        'log_scales': 0.001,
        'cam_m': 1e-4,
        'cam_c': 1e-4,
    }
    print('initializing optimizer', lrs)
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

def set_optimizer_lr(optimizer, variables, args, is_initial_timestep=False):
    lrs = {
        'vertices': args.lr_means3D * variables['scene_radius'],
        'rgb_colors': args.lr_colors,
        'logit_opacities': 0.05 if is_initial_timestep else 0,
        'log_scales': 0.001 if is_initial_timestep else 0,
        'cam_m': 1e-4 if is_initial_timestep else 0,
        'cam_c': 1e-4 if is_initial_timestep else 0,
    }
    print('set optimizer lr', lrs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lrs[param_group['name']]
    return optimizer

def params2rendervar(params, variables):
    vertices = params['vertices']
    faces = variables['faces']
    face_normals = compute_face_normals(vertices, faces)
    means3D = compute_face_barycenters(vertices, faces)
    rotations = compute_q_from_faces(vertices, faces, face_normals)
    variables['unnorm_rotations'] = rotations.detach()
    variables['means3D'] = means3D.detach()
    rendervar = {
        'means3D': means3D,
        'colors_precomp': params['rgb_colors'],
        'rotations': rotations,
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(params['log_scales']),   # using the same scale for all 3 channels (sphere)
        'means2D': torch.zeros_like(means3D, requires_grad=True, device="cuda") + 0,
    }
    return rendervar


def get_loss(params, curr_data, variables, is_initial_timestep, args):
    losses = {}

    rendervar = params2rendervar(params, variables)
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)

    curr_id = curr_data['id']
    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]

    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    losses['scale'] = rendervar['scales'][:, -1].mean()

    face_normals = compute_face_normals(params['vertices'], variables['faces'])
    face_neighbors = variables['face_neighbors']
    neighbor_normals = face_normals[face_neighbors] # (N, k, 3), k=3

    normal_dot = face_normals.unsqueeze(1) * neighbor_normals
    normal_dot = normal_dot.sum(-1) # (N, k)
    norm_mean = normal_dot.mean(-1) # (N,)
    losses['normal'] = (norm_mean - 1.0).abs().mean()

    # we want opacity to be high for the gaussians
    losses['opacity'] = (1 - rendervar['opacities']).mean()

    means3D = compute_face_barycenters(params['vertices'], variables['faces'])
    neighbor_pts = means3D[face_neighbors]
    curr_offset = neighbor_pts - means3D[:, None]
    curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
    losses['iso'] = weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist"], variables["neighbor_weight"])

    # area loss, to prevent degenerate triangles
    vertices = params['vertices']
    faces = variables['faces']
    face_area = compute_face_areas(vertices, faces)
    gaussian_area = rendervar['scales'][:, 0] * rendervar['scales'][:, 1] * torch.pi

    losses['area'] = torch.abs(face_area - gaussian_area).mean()

    mean_area = face_area.mean()
    losses['eq_faces_weight'] = (face_area - mean_area).abs().mean()

    smplx_v = curr_data['smplx_v']
    smplx_vn = curr_data['smplx_vn']


    if 'cloth_v_idx' in variables.keys():
        cloth_vertices = vertices[variables['cloth_v_idx']]
        losses['collision_l'] = collision_penalty(cloth_vertices, smplx_v, smplx_vn,
                                                  return_average=True)
    else:
        losses['collision_l'] = 0.0

    if not is_initial_timestep:
        losses['soft_col_cons'] = l1_loss_v2(params['rgb_colors'], variables["prev_col"])


    loss_weights = {'im': args.img_weight, 'rigid': 0.0, 'rot': .0, 'iso': args.iso_weight, 'floor': 0.0,
                    'soft_col_cons': args.soft_color_weight, 'area': args.area_weight,
                    'scale': 1.0, 'normal': args.normal_weight, 'opacity': args.opacity_weight,
                    'collision_l': args.collision_weight, 'collision_s': args.collision_weight, 'eq_faces_weight': args.eq_faces_weight,
                    'smplx_dis': args.smplx_dis_weight}
    loss = sum([loss_weights[k] * v for k, v in losses.items()])
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen

    if USE_WANDB:
        # Log losses
        for k, v in losses.items():
            wandb.log({k: v.mean().item()})
        wandb.log({'loss': loss.item()})

    return loss, variables


def initialize_per_timestep(params, variables, optimizer):
    pts = params['vertices'].detach()
    rot = torch.nn.functional.normalize(variables['unnorm_rotations'])
    new_pts = pts + (pts - variables["prev_pts"])


    prev_inv_rot_fg = rot
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]

    faces = variables['faces']
    fg_pts = compute_face_barycenters(pts, faces)

    prev_offset = fg_pts[variables["face_neighbors"]] - fg_pts[:, None]
    variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach()
    variables['prev_offset'] = prev_offset.detach()
    variables["prev_col"] = params['rgb_colors'].detach()
    variables["prev_pts"] = pts.detach().clone()
    variables["prev_rot"] = rot.detach()

    if 'cloth_v_idx' in variables.keys():
        cloth_v_idx = variables['cloth_v_idx']
        print('update cloth vertices using inertia, cloth_v_idx: ', cloth_v_idx.shape)
        pts[cloth_v_idx] = new_pts[cloth_v_idx]
    new_params = {'vertices': pts.detach()}

    params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def initialize_post_first_timestep(params, variables, optimizer):
    vertices = params['vertices'].detach()
    faces = variables['faces'].detach()
    normals = compute_face_normals(vertices, faces)
    rots = compute_q_from_faces(vertices, faces, normals)

    variables["prev_pts"] = vertices
    variables["prev_rot"] = torch.nn.functional.normalize(rots).detach()
    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c']
    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0
    return variables

def resume_timestep(params, variables, args):
    ori_params = dict(np.load(os.path.join('./output', args.exp_name, args.save_name, 'params_{}.npz'.format(args.start_idx))))
    resume_params = dict(np.load(os.path.join('./output', args.exp_name, args.save_name, 'params_{}.npz'.format(args.resume_t))))
    ori_params.update(resume_params)
    params['vertices'] = torch.from_numpy(ori_params['vertices']).float().cuda()
    params['log_scales'] = torch.from_numpy(ori_params['log_scales']).float().cuda()
    params['logit_opacities'] = torch.from_numpy(ori_params['logit_opacities']).float().cuda()
    params['cam_m'] = torch.from_numpy(ori_params['cam_m']).float().cuda()
    params['cam_c'] = torch.from_numpy(ori_params['cam_c']).float().cuda()
    params['rgb_colors'] = torch.from_numpy(ori_params['rgb_colors']).float().cuda()
    params = {k: v.requires_grad_() for k, v in params.items()}

    vertices = params['vertices'].detach()
    faces = variables['faces'].detach()
    normals = compute_face_normals(vertices, faces)
    rots = compute_q_from_faces(vertices, faces, normals)

    variables["prev_pts"] = vertices
    variables["prev_rot"] = torch.nn.functional.normalize(rots).detach()

    return params, variables

def report_progress(params, variables, data, i, progress_bar, every_i=100):
    if i % every_i == 0:
        im, _, _, = Renderer(raster_settings=data['cam'])(**params2rendervar(params, variables))
        curr_id = data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]

        psnr = calc_psnr(im, data['im']).mean()

        gaussians_num_c = len(params['rgb_colors'])

        progress_bar.set_postfix({"train img PSNR": f"{psnr:.{7}f}", "n_gaussians": gaussians_num_c})
        progress_bar.update(every_i)
        if USE_WANDB:
            wandb.log({'psnr': psnr.item(), 'gaussians_num': gaussians_num_c})
            if i == 2900:
                # log images
                print("logging images")
                save_pred = im.detach().cpu().numpy().squeeze().transpose(1,2,0)
                save_gt = (data['im']).detach().cpu().numpy().squeeze().transpose(1,2,0)
                save_img = np.zeros((save_pred.shape[0], save_pred.shape[1] * 2, save_pred.shape[2]))
                save_img[:, :save_pred.shape[1], :] = save_pred
                save_img[:, save_pred.shape[1]:save_pred.shape[1]*2, :] = save_gt

                save_img_np = (save_img * 255).astype(np.uint8)
                save_img_np = np.clip(save_img_np, 0, 255)
                save_img_np = cv2.cvtColor(save_img_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(wandb.run.dir, "image_frame_{}.png".format(data['t'])), save_img_np)

                wandb.log({"image frame {} step {}".format(data['t'], i): [wandb.Image(save_img, caption="cloth pred/gt; full img pred/gt")]})


def train(seq, args):
    exp = args.exp_name
    md = json.load(open(f"./data/{seq}/cam_info.json", 'r'))
    start_idx = args.start_idx
    num_timesteps = args.num_frames
    params, variables = initialize_params(seq, md, args.obj_name, args.cloth_name)
    optimizer = initialize_optimizer(params, variables, args)

    lbs_deformer = SmplxDeformer(gender=args.smplx_gender)

    resume_idx = start_idx if not args.resume else args.resume_t
    if args.resume:
        print("resuming from timestep {}".format(resume_idx))
        params, variables = resume_timestep(params, variables, args)
        optimizer = initialize_optimizer(params, variables, args)
    beta = None
    for t in range(resume_idx, resume_idx + num_timesteps):
        dataset = get_dataset(t, md, seq, args.data_path, downsample_view=args.downsample_view)
        todo_dataset = []
        is_initial_timestep = (t == resume_idx)
        if not is_initial_timestep:
            optimizer = set_optimizer_lr(optimizer, variables, args, is_initial_timestep)
            params, variables = initialize_per_timestep(params, variables, optimizer)

        smplx_param_t = dataset[0]['smplx_param']
        for k, v in smplx_param_t.items():
            if k == 'latent':
                v.requires_grad = True
            else:
                v.requires_grad = False
        if not is_initial_timestep:
            smplx_param_t['beta'] = beta.detach()
            smplx_param_t['beta'].requires_grad = False
        else:
            smplx_param_t['beta'].requires_grad = True

        if is_initial_timestep:
            optimizer_smplx = torch.optim.Adam([v for k, v in smplx_param_t.items() if v.requires_grad], lr=args.lr_smplx)
        else:
            optimizer_smplx = torch.optim.Adam([v for k, v in smplx_param_t.items() if v.requires_grad], lr=args.lr_smplx)
        _, smplx_f, _ = read_obj('./data/{}/smplx_fitted/{}.obj'.format(seq, str(t).zfill(6)))
        smplx_f = torch.from_numpy(smplx_f).cuda().long()

        num_iter_per_timestep = 10000 if is_initial_timestep else 3000
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        for i in range(num_iter_per_timestep):
            curr_data = get_batch(todo_dataset, dataset)
            smplx_out = lbs_deformer.smplx_forward(smplx_param_t)
            curr_data['smplx_v'] = smplx_out.vertices.squeeze()
            curr_data['smplx_vn'] = compute_vertex_normals(smplx_out.vertices.squeeze(), smplx_f)
            loss, variables = get_loss(params, curr_data, variables, is_initial_timestep, args)
            loss.backward()
            with torch.no_grad():
                report_progress(params, variables, dataset[6], i, progress_bar)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_smplx.step()
                optimizer_smplx.zero_grad(set_to_none=True)
        progress_bar.close()
        faces = variables["faces"]
        params_ = params
        params_["faces"] = faces

        output_params = params2cpu(params_, is_initial_timestep)
        smplx_save_path = f"./output/{args.exp_name}/{args.save_name}/smplx"
        os.makedirs(smplx_save_path, exist_ok=True)
        torch.save(smplx_param_t, os.path.join(smplx_save_path, f"{str(t).zfill(6)}.pth"))
        lbs_deformer.export(smplx_param_t, os.path.join(smplx_save_path, f"{str(t).zfill(6)}.obj"))

        if is_initial_timestep:
            variables = initialize_post_first_timestep(params, variables, optimizer)
            beta = smplx_param_t['beta']
        # transform the mesh to the new pose
        with torch.no_grad():
            # save mesh
            save_path = f"./output/{args.exp_name}/{args.save_name}"
            os.makedirs(save_path, exist_ok=True)

            print("saving mesh to {}".format(os.path.join(save_path, f"mesh_cloth_{t}.obj")))
            lbs_deformer.save_obj(os.path.join(save_path, f"mesh_cloth_{t}.obj"),
                                  params['vertices'].detach().cpu().numpy().squeeze(),
                                  faces.detach().cpu().numpy().squeeze())

            human_v = params['vertices'].detach().clone()  # (V, 3)
            v_idx = torch.arange(human_v.shape[0]).cuda().long()
            if 'cloth_v_idx' in variables.keys():
                cloth_v_idx = variables['cloth_v_idx']
                v_idx = v_idx[~torch.isin(v_idx, cloth_v_idx)]
                print('human v idx', v_idx.shape)
            if len(v_idx) == 0:
                continue
            human_v = human_v[v_idx]
            human_v = human_v.unsqueeze(0)
            smplx_param0 = dataset[0]['smplx_param']
            smplx_param1 = dataset[0]['smplx_param_1']
            smplx_param0['beta'] = beta
            smplx_param1['beta'] = beta
            smplx = lbs_deformer.smplx_forward(smplx_param0)
            smplx1 = lbs_deformer.smplx_forward(smplx_param1)
            t_human_v, transform_matrix, lbs_w = lbs_deformer.transform_to_t_pose(human_v, smplx, smplx_param0['trans'], smplx_param0['scale'])

            t_human_v = t_human_v.squeeze().unsqueeze(0)
            t_human_v1, transform_matrix1 = lbs_deformer.transform_to_pose(t_human_v, lbs_w, smplx1, smplx_param1['trans'], smplx_param1['scale'])

            params['vertices'][v_idx] = t_human_v1.squeeze()

        os.makedirs(f"./output/{args.exp_name}/{args.save_name}", exist_ok=True)
        np.savez(f"./output/{args.exp_name}/{args.save_name}/params_{t}", **output_params)

if __name__ == "__main__":
    import argparse

    # wandb utils
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_proj', type=str, default='human4d')
    parser.add_argument('--wandb_entity', type=str, default='dress_avatar')
    parser.add_argument('--wandb_name', type=str, default='a1_s1')

    parser.add_argument('--exp_name', type=str, default='exp1_cloth')
    parser.add_argument('--seq', type=str, default='a1_s1')
    parser.add_argument('--start_idx', type=int, default=460)
    parser.add_argument('--num_frames', type=int, default=200)
    parser.add_argument('--save_name', type=str, default='a1_s1')
    parser.add_argument('--obj_name', type=str, default='FrameRec000460.obj')
    parser.add_argument('--cloth_name', type=str, default='cloth_sim.obj')
    parser.add_argument('--data_path', type=str, default='./data/ActorsHQ/Actor01/Sequence1')

    parser.add_argument('--lr_means3D', type=float, default=0.00004)
    parser.add_argument('--lr_colors', type=float, default=0.0025)
    parser.add_argument('--lr_smplx', type=float, default=0.0001)


    parser.add_argument('--normal_weight', type=float, default=0.1)
    parser.add_argument('--soft_color_weight', type=float, default=0.005)
    parser.add_argument('--opacity_weight', type=float, default=0.05)
    parser.add_argument('--iso_weight', type=float, default=20)
    parser.add_argument('--area_weight', type=float, default=50)
    parser.add_argument('--collision_weight', type=float, default=10)
    parser.add_argument('--eq_faces_weight', type=float, default=1000)
    parser.add_argument('--img_weight', type=float, default=1)
    parser.add_argument('--smplx_dis_weight', type=float, default=1)
    parser.add_argument('--downsample_view', type=int, default=1)
    parser.add_argument('--smplx_gender', type=str, default='neutral')


    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_t', type=int, default=849)



    args = parser.parse_args()

    USE_WANDB = args.wandb

    if USE_WANDB:
        save_dir = os.path.join('output', args.exp_name, args.save_name)
        os.makedirs(save_dir, exist_ok=True)
        wandb.init(dir=save_dir, project=args.wandb_proj, name=args.wandb_name,
                   entity=args.wandb_entity, config=args)
        print("wandb initialized {}".format(wandb.run.name))

    sequence = args.seq

    train(sequence, args)