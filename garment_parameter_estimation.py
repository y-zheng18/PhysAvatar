from utils.io_utils import read_obj, save_obj
from codim_ipc_finite_diff import FDM
import os
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    import argparse

    # wandb utils
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_proj', type=str, default='PhysAvatar')
    parser.add_argument('--wandb_entity', type=str, default='dressed_avatar')
    parser.add_argument('--wandb_name', type=str, default='a1_s1')

    parser.add_argument('--exp_name', type=str, default='phys_param_estimation')
    parser.add_argument('--train_dir', type=str, default='./output/exp1_cloth/a1_s1_460_200')
    parser.add_argument('--fix_idx_name', type=str, default='dress_v.txt')
    parser.add_argument('--seq', type=str, default='a1_s1')

    parser.add_argument('--delta_membEMult', type=float, default=0.05)
    parser.add_argument('--delta_bendEMult', type=float, default=0.05)
    parser.add_argument('--delta_density', type=int, default=5)
    parser.add_argument('--per_sim_cpu_num', type=int, default=4)
    parser.add_argument('--frame_num', type=int, default=24)
    parser.add_argument('--start_frame', type=int, default=460)
    parser.add_argument('--max_iters', type=int, default=100)

    args = parser.parse_args()

    # prepare input data
    cloth_v, cloth_f = read_obj(os.path.join(args.train_dir, 'extracted_cloth', f'{str(args.start_frame).zfill(4)}.obj'))

    fixed_idx = np.loadtxt(f'./data/{args.seq}/{args.fix_idx_name}', delimiter=',').astype(np.int32)
    all_indices = np.arange(cloth_v.shape[0])

    non_fixed_idx = np.setdiff1d(all_indices, fixed_idx)
    new_indices = np.concatenate([fixed_idx, non_fixed_idx], axis=0)
    old_to_new_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(new_indices)}

    cloth_v = cloth_v[new_indices]
    cloth_f = np.array([[old_to_new_mapping[i] for i in face] for face in cloth_f])

    cloth_save_root = f'./sim_input/{args.seq}_{args.start_frame}_{args.frame_num}_cloth'
    os.makedirs(cloth_save_root, exist_ok=True)
    save_obj(os.path.join(cloth_save_root, 'dress_reorder.obj'), cloth_v, cloth_f + 1)

    smplx_save_root = f'./sim_input/{args.seq}_{args.start_frame}_{args.frame_num}_smplx'
    fixed_idx_save_path = os.path.join(cloth_save_root, 'garment_bd')
    os.makedirs(fixed_idx_save_path, exist_ok=True)
    os.makedirs(smplx_save_root, exist_ok=True)

    smplx_root = os.path.join(args.train_dir, 'smplx')
    smplx_v, smplx_f = read_obj(os.path.join(smplx_root, f'{str(args.start_frame).zfill(6)}.obj'))

    combined_v = np.concatenate([cloth_v, smplx_v], axis=0)
    combined_f = np.concatenate([cloth_f, smplx_f + cloth_v.shape[0]], axis=0)
    save_obj(os.path.join(cloth_save_root, 'drape_reorder.obj'), combined_v, combined_f + 1)

    for i in tqdm(range(args.frame_num + 1)):
        smplx_v, _ = read_obj(os.path.join(smplx_root, f'{str(i + args.start_frame).zfill(6)}.obj'))
        save_obj(os.path.join(smplx_save_root, 'shell{}.obj'.format(i)), smplx_v, smplx_f + 1)
        if i:
            cloth_v, _ = read_obj(os.path.join(args.train_dir, 'extracted_cloth', f'{str(i + args.start_frame).zfill(4)}.obj'))
            fixed_pos = cloth_v[fixed_idx]

            with open(os.path.join(fixed_idx_save_path, 'actorhq_smplx_{}.txt'.format(i)), 'w') as f:
                for idx, pos in enumerate(fixed_pos):
                    f.write('v {} {} {}'.format(pos[0], pos[1], pos[2]))
                    if idx != fixed_pos.shape[0] - 1:
                        f.write('\n')
            cloth_v = cloth_v[new_indices]
            save_obj(os.path.join(cloth_save_root, f'dress_reordered_{str(i)}.obj'), cloth_v, cloth_f + 1)


    params = FDM(
        seq_name=args.seq,
        garment_path=f'{args.seq}_{args.start_frame}_{args.frame_num}_cloth',
        smplx_path=f'{args.seq}_{args.start_frame}_{args.frame_num}_smplx',
        delta_membEMult=args.delta_membEMult,
        delta_bendEMult=args.delta_bendEMult,
        delta_density=args.delta_density,
        per_sim_cpu_num=args.per_sim_cpu_num,
        frame_num=args.frame_num,
        num_boundary_points=fixed_idx.shape[0],
        use_wandb=args.wandb,
        project=args.wandb_proj,
        entity=args.wandb_entity,
        max_iters=args.max_iters
    )

