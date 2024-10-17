# PhysAvatar PBR (Physically Based Inverse Rendering) Module

This module contains scripts for inverse rendering and optimization of 3D human avatars using Mitsuba 3.

## Usage

To run the inverse rendering process, follow these steps using the ActorsHQ dataset (Actor 1, Sequence 1) as an example:

1. Prepare input data:
   a. Run `export_blender_ply.py` to generate the initial ".ply" file:
      ```bash
      python export_blender_ply.py --mesh_path ./FrameRec000460.obj --npz_path ../output/actor01/seq01/ --name a1/a1s1 --output_dir ./data/a1/
      ```
   b. Generate "a1s1_uv.png" and "a1s1_uv.obj":
      - Follow the UV unwrapping tutorial in Blender [[link to tutorial](https://docs.google.com/presentation/d/e/2PACX-1vTyl0x6Df6o_MFkzuAa_yYsadPJmw5F8NZkjYCFO2zGFVgqggbp_mpDCs4vnOYR0ZEKgbhFLxnsnooM/pub?start=false&loop=false&delayms=3000&slide=id.g2fc63c2ad8e_0_44)]
      - Export the UV-mapped mesh as "a1s1_uv.obj"
      - Export the UV layout as "a1s1_uv.png"

   c. Place files in the correct directories:
      - "./data/a1/a1s1_uv.png"
      - "./data/a1/a1s1_uv.obj"

2. Run the inverse rendering script:
   ```bash
   python inverse_rendering.py \
       --exp_name exp1_cloth \
       --save_name a1_s1 \
       --data_path ../data/ActorsHQ \
       --gsplat_path ../output \
       --num_frames 5
   ```
