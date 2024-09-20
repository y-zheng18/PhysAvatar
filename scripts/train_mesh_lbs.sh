python train_mesh_lbs.py --save_name a1_s1_460_200 --downsample_view 1 --num_frames 200 \
 --lr_means3D 0.00004 --lr_colors 0.0025 --lr_smplx 0 \
 --normal_weight 0.1 --iso_weight 20  --area_weight 50 --eq_faces_weight 1000 --collision_weight 10 \
 --obj_name FrameRec000460.obj --cloth_name cloth_sim.obj --data_path ./data/ActorsHQ/Actor01/Sequence1 \
 --start_idx 460 --wandb --wandb_entity xxxx --wandb_name a1_s1_460_200 --wandb_proj PhysAvatar