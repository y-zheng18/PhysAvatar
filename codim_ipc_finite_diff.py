import subprocess
import random
import numpy as np
import os
import glob
from multiprocessing import Pool
import torch


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

    return vertices, indices


def initialize_parameter(param_ranges):
    individual = {param: random.choice(values) for param, values in param_ranges.items()}
    return individual

def perturb_param(param, param_ranges):
    params_list = []
    params_list.append(param)
    param1 = param.copy()
    param2 = param.copy()
    param3 = param.copy()
    param1['membEMult'] = float(param['membEMult']) + 0.05
    param1['membEMult'] = min(param_ranges['membEMult'][-1], param1['membEMult'])
    param2['bendEMult'] = float(param['bendEMult']) + 0.05
    param2['bendEMult'] = min(param_ranges['bendEMult'][-1], param2['bendEMult'])
    param3['density'] = int(param['density']) + 5
    param3['density'] = min(param_ranges['density'][-1], param3['density'])
    params_list.append(param1)
    params_list.append(param2)
    params_list.append(param3)
    return params_list


def evaluate(individual):
    print("===============================================================")
    print("Evaluating individual:", individual)
    command = f"taskset -c {individual['cpu_a']}-{individual['cpu_b']} python3 codim_ipc_sim.py {individual['algI']} {individual['clothI']} {individual['garmentName']} {individual['membEMult']:05f} {individual['bendEMult']:05f} {individual['seqName']} {individual['density']} {individual['frame_num']} {individual['num_boundary_points']}"
    print(command)
    subprocess.run(command, shell=True, capture_output=True, text=True)
    loss = parse_result(individual)
    print("Loss:", loss)
    print("===============================================================")
    return loss


def evaluate_wrapper(args):
    return evaluate(args)


def parse_result(individual):
    output_path = f"sim_output/codim_ipc_sim/{individual['algI']}_{individual['clothI']}_{individual['garmentName']}_{individual['membEMult']:05f}_{individual['bendEMult']:05f}_{individual['seqName']}_{individual['density']}_{individual['frame_num']}_{individual['num_boundary_points']}/"
    gt_path = f"sim_input/{individual['garmentName']}/"
    simulated_obj = glob.glob(output_path + "*.obj")
    gt_objs = glob.glob(gt_path + "*.obj")
    if len(simulated_obj) != len(gt_objs) - 1:
        print("Simulation failed, check CPU affinity")
        return float('inf')
    loss = 0
    for i in range(len(simulated_obj) - 2):
        simulated_v, _ = read_obj(os.path.join(output_path, f"shell{i + 2}.obj"))
        gt_v, _ = read_obj(gt_path + f"dress_reordered_{i + 1}.obj")
        # print(gt_v.shape, simulated_v.shape)
        loss += np.linalg.norm(gt_v - simulated_v[:gt_v.shape[0]], axis=1).mean()
    loss = loss / (len(simulated_obj) - 1)
    return loss



def FDM(seq_name,
        garment_path,
        smplx_path,
        delta_membEMult=0.05,
        delta_bendEMult=0.05,
        delta_density=5,
        per_sim_cpu_num=4,
        frame_num=24,
        num_boundary_points=228,
        max_iters=100,
        use_wandb=False,
        project=None,
        entity=None):
    fixed_param = {
        'algI': '1',
        'clothI': '2',
        'garmentName': garment_path,
        'seqName': smplx_path,
        'frame_num': frame_num,
        'num_boundary_points': num_boundary_points,
    }
    param_ranges = {
        'bendEMult': [0.5, 8],
        'membEMult': [0.5, 10],
        'density': [200, 640]
    }

    max_iterations = max_iters
    best_iters = 0
    best_loss = float('inf')
    best_param = None

    param = {
        'membEMult': 1.0,
        'bendEMult': 1.0,
        'density': 320
    }

    torch_param = {
        'membEMult': torch.tensor(float(param['membEMult']), requires_grad=False),
        'bendEMult': torch.tensor(float(param['bendEMult']), requires_grad=False),
        'density': torch.tensor(float(param['density']), requires_grad=False)
    }
    optimizer = torch.optim.Adam([torch_param['membEMult'], torch_param['bendEMult'], torch_param['density']], lr=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iterations, eta_min=0.04)

    losses = []
    save_dir = os.path.join('output', f'garment_param_estimation_{seq_name}')
    os.makedirs(save_dir, exist_ok=True)
    output_file = open(os.path.join(save_dir, 'output.txt'), 'w')
    if use_wandb:
        import wandb
        wandb.init(dir=save_dir, project=project, name=f'param_optim_{seq_name}',
                   entity=entity)
        print("wandb initialized {}".format(wandb.run.name))
    with Pool() as pool:
        for _ in range(max_iterations):
            new_params = perturb_param(param, param_ranges)
            args_list = [{**fixed_param, **ind} for ind in new_params]
            last_cpu = 0
            for i, args in enumerate(args_list):
                args['cpu_a'] = last_cpu
                args['cpu_b'] = last_cpu + per_sim_cpu_num
                last_cpu = args['cpu_b'] + 1

            # Run evaluate_fitness in parallel
            loss = pool.map(evaluate_wrapper, args_list)
            if float('inf') in loss:
                print("Simulation failed, check CPU affinity")
                return
            gradients = {}
            loss_ori, loss_m_pert, loss_b_pert, loss_d_pert = loss
            gradients['membEMult'] = (loss_m_pert - loss_ori) / delta_membEMult
            gradients['bendEMult'] = (loss_b_pert - loss_ori) / delta_bendEMult
            gradients['density'] = (loss_d_pert - loss_ori) / delta_density

            optimizer.zero_grad()
            torch_param['membEMult'].grad = torch.tensor(gradients['membEMult']).float()
            torch_param['bendEMult'].grad = torch.tensor(gradients['bendEMult']).float()
            torch_param['density'].grad = torch.tensor(gradients['density']).float()


            optimizer.step()
            lr_scheduler.step()
            # After optimizer.step(), enforce parameter bounds in-place
            torch_param['membEMult'].clamp_(min=float(param_ranges['membEMult'][0]),
                                            max=float(param_ranges['membEMult'][-1]))
            torch_param['bendEMult'].clamp_(min=float(param_ranges['bendEMult'][0]),
                                            max=float(param_ranges['bendEMult'][-1]))
            torch_param['density'].clamp_(min=float(param_ranges['density'][0]), max=float(param_ranges['density'][-1]))

            updated_membEMult = torch_param['membEMult'].item()
            updated_bendEMult = torch_param['bendEMult'].item()
            updated_density = int(torch_param['density'].item())

            losses.append(loss_ori)
            if use_wandb:
                wandb.log({'loss': loss_ori, 'membEMult': updated_membEMult, 'bendEMult': updated_bendEMult,
                           'density': updated_density})
            if loss_ori < best_loss:
                best_loss = loss_ori
                best_param = param.copy()
                best_iters = _
            if use_wandb:
                wandb.log(
                    {'best_loss': best_loss, 'best_iters': best_iters, 'best_memEMult': float(best_param['membEMult']),
                     'best_bendEMult': float(best_param['bendEMult']), 'best_density': int(best_param['density'])})
            output_file.write(f"Iteration {_}: Loss: {loss_ori}, Param: {param}\n"
                              f"Best Loss: {best_loss}, Best Iteration: {best_iters}, Best Param: {best_param}\n")
            print('=====================================================================================================')
            print(f"Iteration {_}: Loss: {loss_ori}, Param: {param}\n"
                  f"Best Loss: {best_loss}, Best Iteration: {best_iters}, Best Param: {best_param})")
            print("gradients:", gradients)
            print('=====================================================================================================')

            param['membEMult'] = updated_membEMult
            param['bendEMult'] = updated_bendEMult
            param['density'] = updated_density

            np.savez(os.path.join(save_dir, 'best_param.npz'),
                     membEMult=best_param['membEMult'],
                     bendEMult=best_param['bendEMult'],
                     density=best_param['density'])

    return best_param

