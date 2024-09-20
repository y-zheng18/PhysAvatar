from utils.io_utils import read_obj, save_obj
import os
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    import argparse

    # wandb utils
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=str, default='./output/exp1_cloth/a1_s1_460_200')
    parser.add_argument('--cloth_name', type=str, default='cloth_sim.obj')
    parser.add_argument('--seq', type=str, default='a1_s1')

    args = parser.parse_args()

    mesh_files = [f for f in os.listdir(args.train_dir) if f.endswith('.npz')]

    cloth_v, cloth_f = read_obj(f'./data/{args.seq}/{args.cloth_name}')

    used_v = np.unique(cloth_f)
    used_v = np.sort(used_v)
    mapping = {v: i for i, v in enumerate(used_v)}
    updated_faces = np.array([[mapping[vertex] for vertex in face] for face in cloth_f])

    full_body_save_dir = os.path.join(args.train_dir, 'extracted_full')
    garment_save_dir = os.path.join(args.train_dir, 'extracted_cloth')
    os.makedirs(full_body_save_dir, exist_ok=True)
    os.makedirs(garment_save_dir, exist_ok=True)
    for f in tqdm(mesh_files):
        frame_idx = int(f.split('_')[-1].split('.')[0])
        params = dict(np.load(os.path.join(args.train_dir, f)))
        save_obj(os.path.join(full_body_save_dir, f'{str(frame_idx).zfill(4)}.obj'), params['vertices'], params['faces'] + 1)

        vertices = params['vertices'][used_v]
        save_obj(os.path.join(garment_save_dir, f'{str(frame_idx).zfill(4)}.obj'), vertices, updated_faces + 1)

