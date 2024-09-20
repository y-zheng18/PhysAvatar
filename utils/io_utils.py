import numpy as np

def load_scene_data(seq, exp):
    params = dict(np.load(f"./output/{exp}/{seq}/params.npz"))
    return params

def read_obj(obj_fn):
    with open(obj_fn) as f:
        lines = f.readlines()
        vertices = []
        faces = []
        for line in lines:
            if line.startswith('v '):  # This line describes a vertex
                vertex = line.strip().split()[1:]  # Split the line into components and discard the first one
                vertex = [float(coord) for coord in vertex]  # Convert the strings to floats
                vertices.append(vertex)
            elif line.startswith('f '):  # This line describes a face
                face = line.strip().split()[1:]  # Split the line into components and discard the first one
                face = [int(index.split('/')[0]) - 1 for index in face]  # Convert the vertex indices to integers
                faces.append(face)
        vertices = np.array(vertices)
        faces = np.array(faces).astype(np.int32)
    return vertices, faces

def save_obj(obj_fn, vertices, faces):
    with open(obj_fn, 'w') as f:
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
