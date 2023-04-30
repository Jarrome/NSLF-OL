import sys, os
import importlib
import open3d as o3d
import argparse
import logging
from time import time
import torch

import cv2
import numpy as np

from scipy.spatial.transform import Rotation as R
import pygame

from utils import exp_util, vis_util
from network import utility
from system import map_color as color_map
from utils.ray_cast import RayCaster

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
key_collect = [pygame.K_o, pygame.K_UP, pygame.K_l, pygame.K_DOWN, 
                    pygame.K_LEFT, pygame.K_RIGHT,pygame.K_w,pygame.K_s,pygame.K_a,pygame.K_d]

# engine
pygame.init()
clock = pygame.time.Clock()
display = None


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
    print('load model from', vis_param.args.outdir+'/nslfs.pt')
    vis_param.color_map.load_nslfs(vis_param.args.outdir+'/nslfs.pt')

    print('load_mesh...')
    # mesh
    if vis_param.args.sequence_kwargs['mesh_gt'] == '':
        #mesh = o3d.io.read_triangle_mesh(vis_param.args.outdir+'bnv.ply')#recons.ply')
        mesh = o3d.io.read_triangle_mesh(vis_param.args.outdir+'recons.ply')

    else:
        mesh = vis_param.sequence.gt_mesh

    #mesh = o3d.io.read_triangle_mesh('tmp.ply')
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
    '''
    pdb.set_trace()
    aframe = next(vis_param.sequence)
    calib_matrix = aframe.calib.to_K() # (3,3)
    H, W, _ = aframe.rgb.shape

    # small size 
    '''
    H, W = 240, 320
    calib_matrix = np.eye(3)
    calib_matrix[0,0] = 320
    calib_matrix[1,1] = 240
    calib_matrix[0,2] = 160
    calib_matrix[1,2] = 120


    vis_param.ray_caster = RayCaster(mesh, H, W, calib_matrix)

    display = pygame.display.set_mode((W,H))

    # based on pygame
    snap_folder = vis_param.args.outdir+'/snap/'
    os.makedirs(snap_folder,exist_ok=True)
    print('now open the control..., q to quit')
    
    running = True
    frame_i = 0
    pose_list = []
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                print(pygame.key.name(event.key))
        keys = pygame.key.get_pressed()

        # use 2.1.2, dont 2.3 their key is wierd...
        if np.sum(keys) == 0:
            continue

        st = time()
        # 1. get controls
        for key, delta_dire in zip(key_collect[:4], delta_dire_collect[:4]):
            if keys[key]:
                posi += delta_dire.dot(heading_bias.dot(heading).dot(facing_dire)) * delta_posi
        for key, delta_dire in zip(key_collect[4:], delta_dire_collect[4:]):
            if keys[key]:
                heading = heading.dot(delta_dire)


        pose = np.eye(4)
        pose[:3,:3] = heading_bias.dot(heading)#R.align_vectors(heading.T, heading_init.T)[0].as_matrix()
        pose[:3,(3,)] = posi
        '''
        with open('tmp_pose.npy', 'rb') as f:
            pose = np.load(f)
        '''



        print('1. key:', time()-st)
        # 2. predict
        pts = vis_param.ray_caster.ray_cast(pose) # N,3
        print('2. ray cast:', time()-st)
        invalid_mask = np.isinf(pts)
        pts[invalid_mask] = 0.
        pts[np.isnan(pts)] = 0.
           
        info = (pose, None, None, None)
        color = vis_param.color_map.integrate_keyframe(torch.tensor(pts).cuda().float(), None,info=info) # N,3
        color[invalid_mask] = 0.
        im = color.reshape((vis_param.ray_caster.H,vis_param.ray_caster.W,3)).cpu().detach().numpy()
        print('3. color predict:', time()-st)

        del color
        torch.cuda.empty_cache() 

        # 3. display
        image = im.transpose(1,0,2)*255
        surf = pygame.surfarray.make_surface(image)
        display.blit(surf, (0, 0))
        pygame.display.update()


        print('4. display:', time()-st)
        snap_name = snap_folder+'/%d.png'%frame_i
        pygame.image.save(surf,snap_name)
        frame_i += 1
        '''
        with open('tmp_pose.npy', 'wb') as f:
            np.save(f, pose)
        '''


        ## store the pose
        #pose_list.append(pose)
    '''
    with open('vr_traj.txt','w') as f:
        for i, pose in enumerate(pose_list):
            q = Quaternion(matrix=pose[:3,:3])
            pose_line = '%d %f %f %f %f %f %f %f'%(i+1, 
                    pose[0,3], pose[1,3], pose[2,3],
                    q[1],q[2],q[2],q[0])

            if i > 0:
                pose_line = '\n'+pose_line
            f.write(pose_line)
    '''
    pygame.quit()

