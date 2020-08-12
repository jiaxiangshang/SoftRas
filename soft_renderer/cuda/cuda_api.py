# -*- coding: utf-8 -*-
# @Author  : Jiaxiang Shang
# @Email   : jiaxiang.shang@gmail.com
# @Time    : 8/10/20 6:36 PM

from torch.utils.cpp_extension import load

#self
import os
_curr_path = os.path.abspath(__file__) # /home/..../face
_cur_dir = os.path.dirname(_curr_path) # ./

load_textures_cuda = load(name='load_textures', sources=[os.path.join(_cur_dir, 'load_textures_cuda.cpp'), os.path.join(_cur_dir, 'load_textures_cuda_kernel.cu')])
soft_rasterize_cuda = load(name='soft_rasterize', sources=[os.path.join(_cur_dir, 'soft_rasterize_cuda.cpp'), os.path.join(_cur_dir, 'soft_rasterize_cuda_kernel.cu')])
create_texture_image_cuda = load(name='create_texture_image', sources=[os.path.join(_cur_dir, 'create_texture_image_cuda.cpp'), os.path.join(_cur_dir, 'create_texture_image_cuda_kernel.cu')])
voxelization_cuda = load(name='voxelization', sources=[os.path.join(_cur_dir, 'voxelization_cuda.cpp'), os.path.join(_cur_dir, 'voxelization_cuda_kernel.cu')])