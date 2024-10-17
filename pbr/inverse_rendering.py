import mitsuba as mi
import drjit as dr
import numpy as np
import glob as glob
from mi_opt_util import get_sensors
import os 
import tqdm
import argparse
import cv2
import camera_data as cd

# Set Mitsuba variant for CUDA-accelerated RGB rendering
mi.set_variant("cuda_ad_rgb")

def parse_args():
    """Parse command-line arguments for the inverse rendering script."""
    parser = argparse.ArgumentParser(description="Inverse Rendering Script")
    parser.add_argument("--actor_idx", type=int, default=1, help="Actor index")
    parser.add_argument("--frame_num", type=int, default=30, help="Number of frames")
    parser.add_argument("--total_frames", type=int, default=180, help="Total number of frames")
    parser.add_argument("--start_frame", type=int, default=0, help="Starting frame")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--res", type=int, default=2048, help="Resolution")
    parser.add_argument("--optimize_light", action="store_true", help="Optimize light")
    parser.add_argument("--mode", type=str, default="short", choices=["short", "long"], help="Mode for actor 2")
    parser.add_argument("--finetune", action="store_true", help="Enable finetuning")
    parser.add_argument("--eval_cam", type=int, default=6, help="Evaluation camera index")
    parser.add_argument("--plot", action="store_true", help="Plot images")
    parser.add_argument("--cam_num", type=int, default=160, help="Number of cameras")
    parser.add_argument("--initial_frame", type=int, default=460, help="Initial frame")
    parser.add_argument("--exp_name", type=str, default='exp1_cloth', help="Experiment name")
    parser.add_argument("--save_name", type=str, default='a1_s1', help="Save name")
    parser.add_argument('--data_path', type=str, default='../data/ActorsHQ', help="Path to data")
    parser.add_argument('--gsplat_path', type=str, default='../output', help="Path to gsplat output - output from train_mesh_lbs.py")

    
    return parser.parse_args()

def main(args):
    """Main function to run the inverse rendering process."""
    # Generate list of camera numbers
    cam_num_list = list(range(args.cam_num))
    print(cam_num_list)   

    # Set up paths for actor data
    actor = f"a{args.actor_idx}"
    actor_ = f"Actor0{args.actor_idx}"
    obj_path = f"./data/{actor}_s1/{actor}s1_uv.obj"
    uv_path = f"./data/{actor}_s1/{actor}s1_uv.png"

    # Downsample UV texture if needed
    texture_img = cv2.imread(uv_path)
    texture_img = cv2.resize(texture_img, (args.res, args.res), interpolation=cv2.INTER_AREA)
    uv_path = uv_path.replace(".png", f"_{args.res}.png")
    cv2.imwrite(uv_path, texture_img)

    # Create output folder
    folder = f"./data/{actor}/{actor}s1_uv_pbr_{args.frame_num}_{int(args.optimize_light)}_{len(cam_num_list)}_{args.batch}_{args.res}/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Set path for output from train_mesh_lbs.py
    path = f"{args.gsplat_path}/{args.exp_name}/{args.save_name}/"

    # Load camera calibration data
    datapath = f"{args.data_path}/{actor_}/Sequence1"
    try:
        cameras = cd.read_calibration_csv(f"{datapath}/4x/calibration_4x.csv")
    except:
        cameras = cd.read_calibration_csv(f"{datapath}/4x/calibration.csv")
        # for i in cam_num_list:
        #     cameras[i] = cameras[i].get_downscaled_camera(4)
    
    # Load vertices and faces
    paths = sorted(glob.glob(path + "*.npz"))
    if args.total_frames is None:
        args.total_frames = len(paths)
    vertices = []
    for i in range(len(paths)):
        data_ = np.load(paths[i])
        vertices.append(data_["vertices"])
        if i == 0:
            face = data_["faces"]

    # Process face order
    face_order = face.reshape(-1)
    _, idx = np.unique(face_order, return_index=True)
    order = face_order[np.sort(idx)]

    # Get all sensors
    sensors_list = [get_sensors(cameras, i) for i in cam_num_list]

    # Load OBJ file data
    line_vertices_list, face_list, uv_list = [], [], []
    with open(obj_path) as f:
        for line in f:
            if line[:2] == "v ":
                line_vertices_list.append(line)
            elif line[:2] == "f ":
                face_list.append(line)
            elif line[:2] == "vt":
                uv_list.append(line)

    # Prepare data for rendering
    myimage_ref_list = []
    mask_map_list = []  
    vertices_list = [] 
    for j in tqdm.tqdm(range(args.start_frame, args.start_frame + args.total_frames, args.total_frames // args.frame_num)):
        # Create temporary OBJ file
        with open(f"./data/a1s1_smart_uv_dummy_{args.actor_idx}.obj", "w") as f:
            line_vertices_list_ = [f"v {v[0]} {v[1]} {v[2]}\n" for v in vertices[j][order]]
            f.writelines(line_vertices_list_)
            f.writelines(uv_list)
            f.writelines(face_list)
        
        # Load mesh and extract vertex positions
        mesh_dummy_mi = mi.load_dict({
            "type": "obj",
            "filename": f"./data/a1s1_smart_uv_dummy_{args.actor_idx}.obj",
        })
        params_dummy = mi.traverse(mesh_dummy_mi)
        vertices_list.append(np.array(params_dummy['vertex_positions']))
        
        # Load and process images and masks for each camera
        myimage_ref_ = []
        mask_map_ = []  
        for i in tqdm.tqdm(cam_num_list):
            # Load and process RGB image
            mybitmap = mi.Bitmap(f"{datapath}/4x/rgbs/Cam{i+1:03}/Cam{i+1:03}_rgb{args.initial_frame+j:06}.jpg").convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=False)
            myimage_ref = np.array(mybitmap)
            
            # Load and process mask
            mask_map = mi.Bitmap(f"{datapath}/4x/masks/Cam{i+1:03}/Cam{i+1:03}_mask{args.initial_frame+j:06}.png").convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=True)
            mask_map = np.array(mask_map)
            kernel = np.ones((3, 3), np.uint8)
            mask_map = cv2.erode(mask_map, kernel, iterations=5)
            mask_map = cv2.GaussianBlur(mask_map, (5, 5), 0)
            
            # Remove white pixels from the image
            white_index = np.where(myimage_ref.mean(axis=-1) > 0.90)
            myimage_ref[white_index[0], white_index[1], :] = 0
            
            myimage_ref_.append(myimage_ref)
            mask_map_.append(mask_map)
        
        myimage_ref_list.append(myimage_ref_)
        mask_map_list.append(mask_map_)

    # Setup scene
    texture_map_path = sorted(glob.glob(f"{folder}/param/*.npy"))
    if len(texture_map_path) > 0:
        # Load existing texture if available
        texture_bp = np.load(texture_map_path[-1])
        print(f"load texture from {texture_map_path[-1]}")
        mesh = mi.load_dict({
            "type": "obj",
            "filename": obj_path,
            "face_normals": True,
            "material": {
                "type": "twosided",
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "bitmap",
                        "data": np.array(texture_bp),
                        "raw": True,
                    },
                },
            },
        })
        it_start = 100 * len(texture_map_path)
    else:
        # Use initial texture if no existing texture
        mesh = mi.load_dict({
            "type": "obj",
            "filename": obj_path,
            "face_normals": True,
            "material": {
                "type": "twosided",
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "bitmap",
                        "filename": uv_path,
                    },
                },
            },
        })
        it_start = 0

    # Set up the scene with lighting
    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "path"},
        "light": {"type": "constant",
                'radiance': {
                            'type': 'rgb',
                            'value': 1,
                        },
                },
        "wavydisk": mesh,
    })

    # Render initial image
    image_init = mi.render(scene, sensor=sensors_list[0], spp=128)
    mi.util.write_bitmap(f"{folder}output_optimize_init.png", image_init)

    # Create output directories
    for dir_name in ['image', 'param', 'texture', 'image_hrs']:
        os.makedirs(f"{folder}/{dir_name}", exist_ok=True)

    # Setup optimization
    params = mi.traverse(scene)
    key = "wavydisk.bsdf.brdf_0.reflectance.data"
    light_key = "light.radiance.value"
    opt = mi.ad.Adam(lr=0.01)
    opt[key] = params[key]
    if args.optimize_light:
        opt[light_key] = params[light_key]

    params.update(opt)

    def mse(image, myimage_ref, mask_map):
        np_imag = np.array(image)
        white_index = np.where(np_imag.mean(axis=-1) > 0.95)
        mask_white = np.ones_like(mask_map)
        mask_white[white_index[0], white_index[1]] = 0
        return dr.sum(dr.sqr(image * mask_map * mask_white - myimage_ref * mask_map)) / (mask_map).sum()

    # Render initial image to check UV mapping
    spp_val = 10
    image = mi.render(scene, params, sensor=sensors_list[0], spp=3)
    mi.util.write_bitmap(f"{folder}_checkuv.png", image)

    # Main optimization loop
    for it in tqdm.tqdm(range(3000)):    
        # Adjust learning rate
        if it % 500 == 0 and it > 0:
            opt = mi.ad.Adam(lr=0.01 / (2 * (it // 500)))
            opt[key] = params[key]
            if args.optimize_light:
                opt[light_key] = params[light_key]
            params.update(opt)
        
        # If it < 1000, use the first frame, otherwise, randomly select a frame
        frame_index = 0 if it < 1000 else np.random.randint(0, args.frame_num)
        camera_index = np.random.randint(0, args.cam_num)
        
        # Update vertex positions
        vertice_ = vertices_list[frame_index]
        params['wavydisk.vertex_positions'] = dr.cuda.Float(vertice_.reshape(-1).tolist())
        params.update()
        
        # Render image
        image = mi.render(scene, params, sensor=sensors_list[camera_index], spp=int(spp_val))
        image_gt = myimage_ref_list[frame_index][camera_index]
        mask_map = mask_map_list[frame_index][camera_index]
        
        # Calculate loss
        loss = mse(image, image_gt, mask_map)

        # Backpropagate and update parameters
        dr.backward(loss)
        opt.step()

        # Clamp texture values
        opt[key] = dr.clamp(opt[key], 0.0, 1.0)
        params.update(opt)
            
        print(f"Iteration {it:02d}; loss = {loss}", end='\r')
        
        # Save intermediate results
        if (it + 1) % 100 == 0 and args.plot:
            mi.util.write_bitmap(f"{folder}/image/image{it}_{camera_index}_{frame_index}.png", image * mask_map)
            mi.util.write_bitmap(f"{folder}/image/image_gt_{camera_index}_{frame_index}.png", image_gt * mask_map)
        if (it + 1) % 100 == 0:
            print('\nOptimization complete.')
            color = np.array(params[key])
            np.save(f"{folder}/param/params_optimize_{it:06}.npy", color)
            if args.optimize_light:
                radiance = np.array(params[light_key])
                np.save(f"{folder}/param/params_optimize_radiance_{it}.npy", radiance)
            if args.plot:
                # Render and evaluate results
                vertice_ = vertices_list[0]
                params['wavydisk.vertex_positions'] = dr.cuda.Float(vertice_.reshape(-1).tolist())
                params.update()
                image = mi.render(scene, params, sensor=sensors_list[args.eval_cam], spp=128)
                image = np.array(image)
                mask = mask_map_list[0][args.eval_cam]
                image_gt = myimage_ref_list[0][args.eval_cam]
                psnr = 10 * np.log10(1 / ((image * mask - image_gt * mask)**2).mean())
                print("psnr", psnr)
                with open(f"{folder}/psnr.txt", "a") as file:
                    file.write(f"{it}: {psnr}\n")
                mi.util.write_bitmap(f"{folder}/image_hrs/image_hrs_{it}_{args.eval_cam}_{frame_index}.png", image)
                color_bit = mi.Bitmap(color)
                mi.util.write_bitmap(f"{folder}/texture/texture_{it:06}.png", color_bit)

if __name__ == "__main__":
    args = parse_args()
    main(args)
