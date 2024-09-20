import sys

sys.path.insert(0, "./Codim-IPC/Python")
import Drivers
from JGSL import *

if __name__ == "__main__":
    sim = Drivers.FEMDiscreteShellBase("double", 3)

    algI = 1
    if len(sys.argv) > 1:
        algI = int(sys.argv[1])

    clothI = 0
    if len(sys.argv) > 2:
        clothI = int(sys.argv[2])

    garmentName = "409_optimized"
    if len(sys.argv) > 3:
        garmentName = sys.argv[3]

    membEMult = 0.1
    if len(sys.argv) > 4:
        membEMult = float(sys.argv[4])

    bendEMult = 1
    if len(sys.argv) > 5:
        bendEMult = float(sys.argv[5])

    seqName = "409_optimized"
    if len(sys.argv) > 6:
        seqName = sys.argv[6]

    density = 472.641509
    if len(sys.argv) > 7:
        density = int(sys.argv[7])

    num_frame = 24
    if len(sys.argv) > 8:
        num_frame = int(sys.argv[8])

    num_boundary_points = 228
    if len(sys.argv) > 9:
        num_boundary_points = int(sys.argv[9])

    thickness_rate = 1
    if len(sys.argv) > 10:
        thickness_rate = float(sys.argv[10])

    counter = sim.add_shell_3D(f"sim_input/{garmentName}/dress_reorder.obj", Vector3d(0, 0, 0), \
                               Vector3d(0, 0, 0), Vector3d(0, 1, 0), 0)

    sim.set_DBC_index(num_boundary_points, Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)
    sim.seqDBC = sim.compNodeRange[-1]
    sim.add_mannequin(f"sim_input/{seqName}/shell0.obj", Vector3d(0, 0, 0), Vector3d(1, 1, 1), \
                      Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)
    print('shell name:', f"sim_input/{seqName}/shell0.obj")
    sim.seqDBCPath = "sim_input/" + seqName
    sim.garmentPath = "sim_input/" + garmentName

    sim.mu = 0.2
    sim.dt = 0.04
    sim.frame_dt = 0.04
    sim.withCollision = True
    sim.mu = 0.2

    # density, E, nu, thickness, initial displacement case
    if algI == 0:
        # iso
        sim.initialize(density, sim.cloth_Ebase_iso[clothI] * membEMult,
                       sim.cloth_nubase_iso[clothI], sim.cloth_thickness_iso[clothI] * thickness_rate, 0)
        sim.bendingStiffMult = bendEMult / membEMult
        sim.kappa_s = Vector2d(1e3, 0)
        sim.s = Vector2d(sim.cloth_SL_iso[clothI], 0)
    elif algI == 1:
        # iso, no strain limit
        sim.initialize(density, sim.cloth_Ebase_iso[clothI] * membEMult,
                       sim.cloth_nubase_iso[clothI], sim.cloth_thickness_iso[clothI] * thickness_rate, 0)
        sim.bendingStiffMult = bendEMult   #/ membEMult
        sim.kappa_s = Vector2d(0, 0)
        sim.s = Vector2d(sim.cloth_SL_iso[clothI], 0)
    elif algI == 2:
        # aniso
        sim.initialize(sim.cloth_density[clothI], sim.cloth_Ebase[clothI],  # actually only affect bending
                       0, sim.cloth_thickness[clothI] * thickness_rate, 0)
        sim.bendingStiffMult = bendEMult
        sim.fiberStiffMult = sim.cloth_weftWarpMult[clothI] * membEMult
        sim.inextLimit = sim.cloth_inextLimit[clothI]
        sim.kappa_s = Vector2d(0, 0)

    sim.load_frame("sim_input/" + garmentName + "/drape_reorder.obj")
    sim.frame_num = num_frame
    sim.initialize_OIPC(1e-3, 0)
    print('start simulating...')
    sim.run()
    print('simulation done')
