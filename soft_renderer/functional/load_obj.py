import os

import torch
import numpy as np
from skimage.io import imread

from thirdParty.SoftRas.soft_renderer.cuda.cuda_api import load_textures_cuda

def load_mtl(filename_mtl):
    '''
    load color (Kd) and filename of textures from *.mtl
    '''
    texture_filenames = {}
    colors = {}
    material_name = ''
    with open(filename_mtl) as f:
        for line in f.readlines():
            if len(line.split()) != 0:
                if line.split()[0] == 'newmtl':
                    material_name = line.split()[1]
                if line.split()[0] == 'map_Kd':
                    texture_filenames[material_name] = line.split()[1]
                if line.split()[0] == 'Kd':
                    colors[material_name] = np.array(list(map(float, line.split()[1:4])))
    return colors, texture_filenames


def load_textures(filename_obj, filename_mtl, texture_res):
    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'vt':
            vertices.append([float(v) for v in line.split()[1:3]])
    vertices = np.vstack(vertices).astype(np.float32)

    # load faces for textures
    faces = []
    material_names = []
    material_name = ''
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            if '/' in vs[0] and '//' not in vs[0]:
                v0 = int(vs[0].split('/')[1])
            else:
                v0 = 0
            for i in range(nv - 2):
                if '/' in vs[i + 1] and '//' not in vs[i + 1]:
                    v1 = int(vs[i + 1].split('/')[1])
                else:
                    v1 = 0
                if '/' in vs[i + 2] and '//' not in vs[i + 2]:
                    v2 = int(vs[i + 2].split('/')[1])
                else:
                    v2 = 0
                faces.append((v0, v1, v2))
                material_names.append(material_name)
        if line.split()[0] == 'usemtl':
            material_name = line.split()[1]
    faces = np.vstack(faces).astype(np.int32) - 1
    faces = vertices[faces]
    faces = torch.from_numpy(faces).cuda()
    faces[1 < faces] = faces[1 < faces] % 1

    colors, texture_filenames = load_mtl(filename_mtl)

    textures = torch.ones(faces.shape[0], texture_res**2, 3, dtype=torch.float32)
    textures = textures.cuda()

    #
    for material_name, color in list(colors.items()):
        color = torch.from_numpy(color).cuda()
        for i, material_name_f in enumerate(material_names):
            if material_name == material_name_f:
                textures[i, :, :] = color[None, :]

    for material_name, filename_texture in list(texture_filenames.items()):
        filename_texture = os.path.join(os.path.dirname(filename_obj), filename_texture)
        image = imread(filename_texture).astype(np.float32) / 255.

        # texture image may have one channel (grey color)
        if len(image.shape) == 2:
            image = np.stack((image,)*3, -1)
        # or has extral alpha channel shoule ignore for now
        if image.shape[2] == 4:
            image = image[:, :, :3]

        # pytorch does not support negative slicing for the moment
        image = image[::-1, :, :]
        image = torch.from_numpy(image.copy()).cuda()
        is_update = (np.array(material_names) == material_name).astype(np.int32)
        is_update = torch.from_numpy(is_update).cuda()

        # print(filename_texture)
        # print(is_update)
        # print(faces[50:150])
        # print(image[50:150])
        textures = load_textures_cuda.load_textures(image, faces, textures, is_update)
        #print(textures)
        # with open('/data0/4_Exp/0_mdfsnet/obj_uv.txt', 'w') as f_debug:
        #     is_update = is_update.cpu().detach().numpy()
        #     faces = faces.cpu().detach().numpy()
        #     image = image.cpu().detach().numpy()
        #     textures = textures.cpu().detach().numpy()
        #     for i in range(50, 100):
        #         f_debug.write("ud %d, %f\n" % (i,is_update[i]))
        #         f_debug.write("uv %d, %f, %f\n"%(i, faces[i][0][0], faces[i][0][1]))
        #         f_debug.write("image %d, %f, %f, %f\n"%(i, image[i][100][0], image[i][100][1], image[i][100][2]))
        #         f_debug.write("texture %d, %f, %f, %f"%(i, textures[i][0][0], textures[i][0][1], textures[i][0][2]))
    return textures


def load_obj(filename_obj, normalization=False, load_texture=False, texture_res=4, texture_type='surface'):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """

    assert texture_type in ['surface', 'vertex']

    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = torch.from_numpy(np.vstack(vertices).astype(np.float32)).cuda()

    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = torch.from_numpy(np.vstack(faces).astype(np.int32)).cuda() - 1

    # load textures
    if load_texture and texture_type == 'surface':
        textures = None
        for line in lines:
            if line.startswith('mtllib'):
                filename_mtl = os.path.join(os.path.dirname(filename_obj), line.split()[1])
                textures = load_textures(filename_obj, filename_mtl, texture_res)
        if textures is None:
            raise Exception('Failed to load textures.')
    elif load_texture and texture_type == 'vertex':
        textures = []
        for line in lines:
            if len(line.split()) == 0:
                continue
            if line.split()[0] == 'v':
                textures.append([float(v) for v in line.split()[4:7]])
        textures = torch.from_numpy(np.vstack(textures).astype(np.float32)).cuda()

    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)[0][None, :]
        vertices /= torch.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[0][None, :] / 2

    if load_texture:
        return vertices, faces, textures
    else:
        return vertices, faces
