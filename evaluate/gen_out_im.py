import os, sys
import importlib
import open3d as o3d
import argparse
import logging
from time import time
import torch

import cv2
import numpy as np

from PIL import Image
from scipy.spatial.transform import Rotation as R


sys.path.append('.')
from utils import exp_util, vis_util
from network import utility
from system import map_color as color_map
from utils.ray_cast import RayCaster

from metrics import img2mse, mse2psnr


import pdb


# param
vis_param = argparse.Namespace()
vis_param.n_left_steps = 0
vis_param.args = None
vis_param.mesh_updated = True





# control
heading_bias = None#
heading = None
posi = None

facing_dire = np.array([[0,0,1.]]).T
delta_posi = .1
delta_dire_up = R.from_euler('y', 90, degrees=True).as_matrix()
delta_dire_forward = np.eye(3)
delta_dire_down = R.from_euler('y', -90, degrees=True).as_matrix()
delta_dire_back = -np.eye(3)

delta_degree = 2.
delta_dire_turnleft = R.from_euler('y', -delta_degree, degrees=True).as_matrix()
delta_dire_turnright = R.from_euler('y', delta_degree, degrees=True).as_matrix()
delta_dire_turnup = R.from_euler('x', delta_degree, degrees=True).as_matrix()
delta_dire_turndown = R.from_euler('x', -delta_degree, degrees=True).as_matrix()
delta_dire_turnrollleft = R.from_euler('z', -delta_degree, degrees=True).as_matrix()
delta_dire_turnrollright = R.from_euler('z', delta_degree, degrees=True).as_matrix()




delta_dire_collect = [delta_dire_up, delta_dire_forward, delta_dire_down, delta_dire_back, # for translation of view point 
                        delta_dire_turnleft, delta_dire_turnright, delta_dire_turnup, delta_dire_turndown, delta_dire_turnrollleft, delta_dire_turnrollright] # for rotation of view point

# engine
display = None


def save_image(img_array, filepath):
    assert len(img_array.shape) == 2 or (len(img_array.shape) == 3
                                         and img_array.shape[2] in [3, 4])

    if img_array.dtype != np.uint8:
        img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    Image.fromarray(img_array).save(filepath)



if __name__ == '__main__':
    parser = exp_util.ArgumentParserX()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # Load in network.  (args.model is the network specification)
    #model, args_model = utility.load_model(args.training_hypers, args.using_epoch)
    args.model = None#args_model
    args.mapping = exp_util.dict_to_args(args.mapping)
    args.color_mapping = exp_util.dict_to_args(args.color_mapping)
    args.tracking = exp_util.dict_to_args(args.tracking)

    # Load in sequence.
    seq_package, seq_class = args.sequence_type.split(".")
    sequence_module = importlib.import_module("dataset.production." + seq_package)
    sequence_module = getattr(sequence_module, seq_class)
    vis_param.sequence = sequence_module(**args.sequence_kwargs)
    vis_param.args = args

    # Mapping
    if torch.cuda.device_count() > 1:
        main_device, aux_device = torch.device("cuda", index=0), torch.device("cuda", index=1)
    elif torch.cuda.device_count() == 1:
        main_device, aux_device = torch.device("cuda", index=0), None
    else:
        assert False, "You must have one GPU."

    vis_param.color_map = color_map.DenseIndexedMap(None, args.color_mapping, 10, main_device,
                                        args.run_async, aux_device)
 
    # finished, not save model
    print('load model from', vis_param.args.outdir+'/nerfs.pt')
    vis_param.color_map.load_nerfs(vis_param.args.outdir+'/nerfs.pt')

    print('load_mesh...')
    # mesh
    if vis_param.args.sequence_kwargs['mesh_gt'] == '':
        print("from build mesh")
        mesh = o3d.io.read_triangle_mesh(vis_param.args.outdir+'recons.ply')
    else:
        print('from gt mesh')
        mesh = vis_param.sequence.gt_mesh

    #mesh = o3d.io.read_triangle_mesh('tmp.ply')
    #mesh = o3d.io.read_triangle_mesh(vis_param.args.outdir+'bnv.ply')#recons.ply')

    '''
    import trimesh
    #mesh = trimesh.load('tmp.ply')
    mesh = trimesh.load(vis_param.args.outdir+'recons.ply')
    trimesh.repair.fill_holes(mesh)
    mesh = mesh.as_open3d
    '''


    # pose
    first_iso = vis_param.sequence.first_iso#Isometry(q=Quaternion(array=first_tq[3:]), t=np.array(first_tq[:3]))
    change_pose = first_iso.matrix

    heading_bias = change_pose[:3,:3]#.dot(np.array([[1.,0,0]]).T)
    heading = np.eye(3)
    posi = change_pose[:3,(3,)]



    # raycaster
    aframe = next(vis_param.sequence)
    calib_matrix = aframe.calib.to_K() # (3,3)
    H, W, _ = aframe.rgb.shape
    vis_param.ray_caster = RayCaster(mesh, H, W, calib_matrix)


    #####
    #vis_param.sequence.gt_trajectory


    running = True
    id = 0
    metrics = dict()
    metrics['psnr'] = []


    os.makedirs(os.path.join(vis_param.args.outdir,'image'), exist_ok=True)
    time_lst = []
    try:
      while True:
        pose = vis_param.sequence.gt_trajectory[id].matrix
        print("Frame ",id)
        id += 1
        # 2. predict
        st = time()
        pts = vis_param.ray_caster.ray_cast(pose) # N,3
        print('2. ray cast:', time()-st)
        invalid_mask = np.isinf(pts)
        pts[invalid_mask] = 0.
        info = (pose, None, None, None)
        color = vis_param.color_map.integrate_keyframe(torch.tensor(pts).cuda().float(), None,info=info) # N,3
        color[invalid_mask] = .5
        color = color.reshape((vis_param.ray_caster.H,vis_param.ray_caster.W,3))
        im = color.cpu().detach().numpy()
        record_time = time()-st
        time_lst.append(record_time)
        print('3. color predict:', record_time)
        save_image(im, os.path.join(vis_param.args.outdir,'image','%d.png'%(id-1)))
    except Exception as e:
        print(e)
        print("Rendering speed:", np.mean(time_lst))

