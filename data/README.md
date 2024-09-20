# Data preparation

## Garment segmentation
We suggest to use Blender to extract the garment mesh if you are familiar with the software, 
or you can use [CloSe](https://virtualhumans.mpi-inf.mpg.de/close3dv24/), 
a 3D garment segmentation network.
## Boundry point indices
For simulation purpose, we need to define boundry points that moves with the human body and drives the simulation for the rest of the garment. 
A straight-forward way to obtain the boundry point indices is to import the garment mesh (e.g., `./output/exp1_cloth/a1_s1_460_200/extracted_cloth/*.obj`) to Blender, 
select the boundry points (e.g., points on the top of the dress), and export the indices of the selected points using the following script (copied from [here](https://github.com/V-Sekai/TOOL_cloth_dynamics?tab=readme-ov-file#blender-addon-for-selecting-vertex-indexes)):

```
import bpy

bpy.ops.object.mode_set(mode='OBJECT')
obj = bpy.context.active_object
selected_vertices = [v.index for v in obj.data.vertices if v.select]

print(selected_vertices)
```

## SMPLX fitting
* To be updated