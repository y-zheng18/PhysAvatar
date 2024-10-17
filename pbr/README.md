# PhysAvatar PBR (Physics-Based Inverse Rendering) Module

This module contains scripts for inverse rendering and optimization of 3D human avatars using Mitsuba 3.

## Usage

To run the inverse rendering process, follow these steps using the ActorsHQ dataset (Actor 1, Sequence 1) as an example:

0. Install the required dependencies:
   ```bash
   mamba env create -f environment_render.yml
   conda activate mi_renderer
   ```
1. Prepare input data:

   a. Run `export_blender_ply.py` to generate the initial ".ply" file:
      ```bash
      python pbr/export_blender_ply.py --mesh_path ./data/a1_s1/FrameRec000460.obj --npz_path ./output/exp1_cloth/a1_s1_460_200 --name a1_s1 --output_dir ./data/a1_s1
      ```
   b. Generate "a1s1_uv.png" and "a1s1_uv.obj":
      - Follow the UV unwrapping tutorial in Blender [[link to tutorial](https://docs.google.com/presentation/d/e/2PACX-1vTyl0x6Df6o_MFkzuAa_yYsadPJmw5F8NZkjYCFO2zGFVgqggbp_mpDCs4vnOYR0ZEKgbhFLxnsnooM/pub?start=false&loop=false&delayms=3000&slide=id.g2fc63c2ad8e_0_44)]
      - Export the UV-mapped mesh as "a1s1_uv.obj"
      - Export the UV layout as "a1s1_uv.png"

   c. Place files in the correct directories:
      - "./data/a1_s1/a1s1_uv.png"
      - "./data/a1_s1/a1s1_uv.obj"

   d. alternatively, you can download the prepared data from [this link](https://drive.google.com/file/d/1-w4kl0BxrKT6d8IRr2GO8Uc-bNX3r-n_/view?usp=sharing)

3. Run the inverse rendering script:
   ```bash
   python pbr/inverse_rendering.py \
       --exp_name exp1_cloth \
       --save_name a1_s1_460_200 \
       --data_path ./data/ActorsHQ/ \
       --gsplat_path ./output \
       --frame_num 5 \
       --plot
   ```
Optimized texture maps will be saved in the `./data/a1/a1s1_uv_pbr_5_0_160_1_2048/texture` directory. The texture map can be used for rendering using Mitsuba, Blender, and other rendering engines.