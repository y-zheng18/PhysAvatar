import mitsuba as mi
import numpy as np

def read_obj(obj_path):
    # Read the obj file
    with open(obj_path, "r") as f:
        lines = f.readlines()
    # Extract vertices and faces
    vertices = []
    faces = []
    for line in lines:
        if line.startswith("v "):
            vertices.append(list(map(float, line.split()[1:])))
        elif line.startswith("f "):
            faces.append(list(map(int, line.split()[1:])))
    vertices = np.array(vertices)
    faces = np.array(faces) 
    return vertices, faces-1

def get_sensors(cameras, index, translation_offset=[0, 0, 0], resolution_factor=[1, 1], downsample=1):
    camera = cameras[index].get_downscaled_camera(downsample)
    camera.width = int(camera.width * resolution_factor[0])
    camera.height = int(camera.height * resolution_factor[1])

    translation = camera.extrinsic_matrix_cam2world()[:3, 3] + translation_offset
    rotation_matrix = camera.rotation_matrix_cam2world()

    look_at = np.matmul(rotation_matrix, [0, 0, 1]) + translation
    up = np.matmul(rotation_matrix, [0, -1, 0])

    intrinsic_matrix = camera.intrinsic_matrix()
    fov = (2 * np.arctan2(camera.height, 2 * float(intrinsic_matrix[1][1]))) * 180 / np.pi

    principal_point_offset_x = (camera.principal_point[0] * camera.width - camera.width / 2) / camera.width * -1
    principal_point_offset_y = (camera.principal_point[1] * camera.height - camera.height / 2) / camera.height * -1

    return mi.load_dict({
        "type": "perspective",
        "to_world": mi.ScalarTransform4f.look_at(origin=translation.tolist(), target=look_at.tolist(), up=up.tolist()),
        "principal_point_offset_x": principal_point_offset_x,
        "principal_point_offset_y": principal_point_offset_y,
        "fov": fov,
        "fov_axis": "y",
        "film": {
            "width": camera.width,
            "height": camera.height,
            "type": "hdrfilm",
            "component_format": "float32",
        },
    })

def interpolate_vertex_colors(faces, face_colors, num_v):
    # Assuming face_colors is a list of RGB colors

    # Initialize sum of colors and count for each vertex
    vertex_color_sum = {}
    vertex_count = {}

    # Iterate over each face and its color
    for face, color in zip(faces, face_colors):
        for vertex in face:
            if vertex not in vertex_color_sum:
                vertex_color_sum[vertex] = np.zeros(3)
                vertex_count[vertex] = 0

            # Add the face color to the vertex color sum
            vertex_color_sum[vertex] += np.array(color)
            vertex_count[vertex] += 1

    vertex_colors = []
    for i in range(num_v):
        if i in vertex_count.keys():
            vertex_colors.append((vertex_color_sum[i] / vertex_count[i]).tolist())
        else:
            vertex_colors.append([0,0,0])
            
    vertex_colors = np.array(vertex_colors)
    print(vertex_colors.shape)
    return vertex_colors