from utils.smplx_deformer import SmplxDeformer
import os
import numpy as np
import tqdm
import torch
import subprocess
import pytorch3d
from scipy.spatial.transform import Rotation

if __name__ == "__main__":
    import argparse

    # wandb utils
    parser = argparse.ArgumentParser()

    parser.add_argument('--cloth_name', type=str, default='cloth_sim.obj')
    parser.add_argument('--obj_name', type=str, default='FrameRec000460.obj')
    parser.add_argument('--body_name', type=str, default='body_sim.obj')
    parser.add_argument('--fix_idx_name', type=str, default='dress_v.txt')
    parser.add_argument('--seq', type=str, default='a1_s1')
    parser.add_argument('--smplx_gender', type=str, default='neutral')
    parser.add_argument('--smplx_param_name', type=str, default='000460.pth')
    parser.add_argument('--param_path', type=str, default='./output/garment_param_estimation_a1_s1/best_param.npz')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--frame_num', type=int, default=340)
    parser.add_argument('--speed_up_ratio', type=int, default=2)
    parser.add_argument('--motion_path', type=str, default='./data/AMASS/MoSh/50020/shake_hips_stageii.npz')
    parser.add_argument('--max_sim_time', type=int, default=8, help='max simulation time in hours')
    parser.add_argument('--save_dir', type=str, default='./output/sim_actorhq_amass/')

    args = parser.parse_args()

    # prepare input data
    lbs_deformer = SmplxDeformer(gender=args.smplx_gender)

    # read motion data and convert to our smplx format
    smplx_param = np.load(args.motion_path)
    src_param = torch.load(f'./data/{args.seq}/smplx_fitted/{args.smplx_param_name}', map_location=lbs_deformer.device)
    device = lbs_deformer.device

    src_param.pop('latent')

    smplx_save_path = os.path.join(args.save_dir, 'amass_smplx')
    os.makedirs(smplx_save_path, exist_ok=True)
    for i in tqdm.trange(min(len(smplx_param['poses']) // args.speed_up_ratio, args.frame_num)):
        with torch.no_grad():
            src_param['body_pose'] = torch.from_numpy(smplx_param['pose_body'][int(i * args.speed_up_ratio)]).to(device).unsqueeze(0).float()
            trans = smplx_param['trans'][int(i * args.speed_up_ratio)]
            orient = smplx_param['root_orient'][int(i * args.speed_up_ratio)]
            R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            trans = R @ trans
            orient_R = R @ Rotation.from_rotvec(orient).as_matrix()
            orient = Rotation.from_matrix(orient_R).as_rotvec()
            src_param['trans'] = torch.from_numpy(trans).to(device).unsqueeze(0).float()
            src_param['orient'] = torch.from_numpy(orient).to(device).unsqueeze(0).float()

            lbs_deformer.export(src_param, f'{smplx_save_path}/{str(i + 10).zfill(4)}.obj')
            torch.save(src_param, f'{smplx_save_path}/{str(i + 10).zfill(4)}.pth')


    # using optimized lbs weights to animate full body
    fullbody_v, fullbody_f = lbs_deformer.read_obj(f'./data/{args.seq}/{args.obj_name}')
    cloth_v, cloth_f = lbs_deformer.read_obj(f'./data/{args.seq}/{args.cloth_name}')
    body_v, body_f = lbs_deformer.read_obj(f'./data/{args.seq}/{args.body_name}')

    cloth_idx = np.unique(cloth_f)
    cloth_mapping = {v: i for i, v in enumerate(cloth_idx)}
    updated_cloth_faces = np.array([[cloth_mapping[vertex] for vertex in face] for face in cloth_f])

    body_idx = np.unique(body_f)
    body_mapping = {v: i for i, v in enumerate(body_idx)}
    updated_body_faces = np.array([[body_mapping[vertex] for vertex in face] for face in body_f])

    fixed_idx = np.loadtxt(f'./data/{args.seq}/{args.fix_idx_name}', delimiter=',').astype(np.int32)
    all_indices = np.arange(len(cloth_idx))

    non_fixed_idx = np.setdiff1d(all_indices, fixed_idx)
    new_indices = np.concatenate([fixed_idx, non_fixed_idx], axis=0)
    old_to_new_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(new_indices)}
    cloth_f = np.array([[old_to_new_mapping[i] for i in face] for face in updated_cloth_faces])

    with torch.no_grad():
        src_v, src_f = lbs_deformer.read_obj(f'./data/{args.seq}/smplx_fitted/{args.smplx_param_name.replace(".pth", ".obj")}')

        lbs_w = np.load(f'./data/{args.seq}/optimized_weights.npy').astype(np.float32)

        smplx_param = torch.load(f'./data/{args.seq}/smplx_fitted/{args.smplx_param_name}',
                                 map_location=lbs_deformer.device)

        smplx = lbs_deformer.smplx_forward(smplx_param)

        t_human_v, transform_matrix, lbs_w = lbs_deformer.transform_to_t_pose(torch.from_numpy(fullbody_v).unsqueeze(0),
                                                                              smplx,
                                                                              smplx_param['trans'],
                                                                              smplx_param['scale'],
                                                                              lbs_w=torch.from_numpy(lbs_w).unsqueeze(
                                                                                  0))
        t_human_v = t_human_v.squeeze().unsqueeze(0)

        pbar = tqdm.trange(args.frame_num + 10)
        pbar.set_description('Generating simulation input...')
        cloth_save_root = os.path.join('./sim_input', f'AMASS_{args.seq}_{args.start_frame}_{args.frame_num}')
        fullbody_save_root = os.path.join('./sim_input', f'AMASS_{args.seq}_{args.start_frame}_{args.frame_num}_full')
        smplx_save_root = os.path.join('./sim_input', f'AMASS_{args.seq}_{args.start_frame}_{args.frame_num}_smplx')
        fixed_idx_save_root = os.path.join(cloth_save_root, 'garment_bd')
        os.makedirs(cloth_save_root, exist_ok=True)
        os.makedirs(fullbody_save_root, exist_ok=True)
        os.makedirs(smplx_save_root, exist_ok=True)
        os.makedirs(fixed_idx_save_root, exist_ok=True)

        smplx_param1 = torch.load(os.path.join(smplx_save_path, '0010.pth'))
        smplx_param1 = {k: v.to(lbs_deformer.device) for k, v in smplx_param1.items()}
        smplx_param0 = smplx_param
        smplx_param0['trans'] = smplx_param1['trans']
        smplx_param0['orient'] = smplx_param1['orient']
        smplx_param0['body_pose'] = pytorch3d.transforms.rotation_conversions.matrix_to_axis_angle(
                lbs_deformer.vposer.decode(smplx_param0['latent']).view(-1, 3, 3)).view(1, -1)
        for i in pbar:
            if i < 10:
                # interpolation
                smplx_param_i = {k: smplx_param1[k] * i / 10 + smplx_param0[k] * (10 - i) / 10 for k in smplx_param1.keys()}
                smplx_i = lbs_deformer.smplx_forward(smplx_param_i)
                human_v_i, transform_matrix1 = lbs_deformer.transform_to_pose(t_human_v, lbs_w, smplx_i,
                                                                               smplx_param_i['trans'],
                                                                               smplx_param_i['scale'])
            else:
                smplx_param_i = torch.load(os.path.join(smplx_save_path, '{}.pth'.format(str(i).zfill(4))),
                                          map_location=lbs_deformer.device)
                smplx_param_i = {k: v.to(lbs_deformer.device) for k, v in smplx_param_i.items()}
                smplx_i = lbs_deformer.smplx_forward(smplx_param_i)
                human_v_i, transform_matrix1 = lbs_deformer.transform_to_pose(t_human_v, lbs_w, smplx_i,
                                                                               smplx_param_i['trans'],
                                                                               smplx_param_i['scale'])

            smplx_i_v = smplx_i.vertices.numpy().reshape(-1, 3)
            human_v_i = human_v_i.numpy().reshape(-1, 3)
            obstacle_v = np.concatenate([smplx_i_v, human_v_i[body_idx]], axis=0)
            obstacle_f = np.concatenate([src_f, updated_body_faces + smplx_i_v.shape[0]], axis=0)
            lbs_deformer.save_obj(os.path.join(smplx_save_root, f'shell{i}.obj'),
                                  obstacle_v,
                                  obstacle_f)

            lbs_deformer.save_obj(os.path.join(fullbody_save_root, f'{str(i).zfill(4)}.obj'),
                                  human_v_i,
                                  fullbody_f)

            cloth_v_i = human_v_i[cloth_idx]

            if i == 0:
                lbs_deformer.save_obj(os.path.join(cloth_save_root, 'dress_reorder.obj'), cloth_v_i[new_indices],
                                      cloth_f)
                combined_v = np.concatenate([cloth_v_i[new_indices], obstacle_v], axis=0)
                combined_f = np.concatenate([cloth_f, obstacle_f + cloth_v_i.shape[0]], axis=0)
                lbs_deformer.save_obj(os.path.join(cloth_save_root, 'drape_reorder.obj'), combined_v, combined_f)
            else:
                fixed_pos = cloth_v_i[fixed_idx]
                with open(os.path.join(fixed_idx_save_root, 'actorhq_smplx_{}.txt'.format(i)), 'w') as f:
                    for idx, pos in enumerate(fixed_pos):
                        f.write('v {} {} {}'.format(pos[0], pos[1], pos[2]))
                        if idx != fixed_pos.shape[0] - 1:
                            f.write('\n')
                cloth_v_i = cloth_v_i[new_indices]
                lbs_deformer.save_obj(os.path.join(cloth_save_root, f'dress_reordered_{str(i)}.obj'), cloth_v_i,
                                      cloth_f)

            pbar.set_postfix_str(f'Frame {i + args.start_frame} done')

        # run simulation
        sim_args = {
            'algI': '1',
            'clothI': '2',
            'garmentName': f'AMASS_{args.seq}_{args.start_frame}_{args.frame_num}',
            'seqName': f'AMASS_{args.seq}_{args.start_frame}_{args.frame_num}_smplx',
            'frame_num': args.frame_num + 10,
            'num_boundary_points': len(fixed_idx),
        }
        optimized_params = np.load(args.param_path)
        sim_args.update(optimized_params)

        command_sim = f"python3 codim_ipc_sim.py {sim_args['algI']}" \
                      f" {sim_args['clothI']} {sim_args['garmentName']} " \
                      f"{sim_args['membEMult']:05f} {sim_args['bendEMult']:05f}" \
                      f" {sim_args['seqName']} {sim_args['density']}" \
                      f" {sim_args['frame_num']} {sim_args['num_boundary_points']}"
        # set max simulation time
        max_sim_time = 60 * 60 * args.max_sim_time  # in seconds
        print(f"Running simulation with command: {command_sim}")
        # run simulation
        try:
            # Run the command with a timeout
            subprocess.run(command_sim, shell=True, timeout=max_sim_time)
        except subprocess.TimeoutExpired:
            print(f"Simulation exceeded the time limit of {args.max_sim_time} hour(s) and was terminated.")

        # save simulation results
        sim_output_path = f"./sim_output/codim_ipc_sim/{sim_args['algI']}_" \
                          f"{sim_args['clothI']}_{sim_args['garmentName']}_" \
                          f"{sim_args['membEMult']:05f}_{sim_args['bendEMult']:05f}_" \
                          f"{sim_args['seqName']}_{sim_args['density']}_{sim_args['frame_num']}_" \
                          f"{sim_args['num_boundary_points']}/"
        save_dir = os.path.join(args.save_dir, f'amass_animation')
        os.makedirs(save_dir, exist_ok=True)
        sim_obj_list = os.listdir(sim_output_path)
        sim_obj_list = [obj for obj in sim_obj_list if obj.endswith('.obj')]

        for i in tqdm.trange(len(sim_obj_list) - 1):
            # reorder the vertices to match the original order
            sim_cloth_v_i, _ = lbs_deformer.read_obj(f'{sim_output_path}/shell{i + 1}.obj')
            sim_cloth_v_i = sim_cloth_v_i[:len(cloth_idx)]

            full_body_v, _ = lbs_deformer.read_obj(f'{fullbody_save_root}/{str(i).zfill(4)}.obj')

            sim_cloth_v_i_ori_order = np.zeros(sim_cloth_v_i.shape)
            sim_cloth_v_i_ori_order[new_indices] = sim_cloth_v_i

            full_body_v[cloth_idx] = sim_cloth_v_i_ori_order
            lbs_deformer.save_obj(os.path.join(save_dir, f'{str(i).zfill(4)}.obj'), full_body_v, fullbody_f)




