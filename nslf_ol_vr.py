import os
import sys
import importlib
import open3d as o3d
import argparse
import logging
import time
import torch
from utils import exp_util, vis_util
from network import utility
import numpy as np
from system import map_surface as surface_map
from system import map_color as color_map

import system.tracker_surface 

from utils import motion_util
from pyquaternion import Quaternion

# vr
from scipy.spatial.transform import Rotation as R
import pygame
from utils.ray_cast import RayCaster
from threading import Thread
from copy import copy
import pdb

vis_param = argparse.Namespace()
vis_param.n_left_steps = 0
vis_param.args = None
vis_param.mesh_updated = True

vis_param.whole_pc = np.zeros((0,3))



def key_continue(vis):
    vis_param.n_left_steps += 20000
    return False


st = None
prev_rgbd = None
pose = None
map_mesh_copy = None
map_mesh_updated = False # a signal shared by refresh() and vr()
def refresh(vis):
    global st, prev_rgbd, pose, pc, map_mesh_copy, map_mesh_updated
    it_st = time.time()
    if vis_param.sequence.frame_id % vis_param.args.integrate_interval == 0:
        st = time.time()
    if vis:
        pass
        # This spares slots for meshing thread to emit commands.
        #time.sleep(0.02)


    if not vis_param.mesh_updated and vis_param.args.run_async:
        st = time.time()
        map_mesh = vis_param.map.extract_mesh(vis_param.args.resolution, 0, extract_async=True)
        if map_mesh is not None:
            map_mesh_updated = True
            map_mesh_copy = copy(map_mesh)
            vis_param.mesh_updated = True
            o3d.io.write_triangle_mesh(args.outdir+'/recons.ply', map_mesh)
            print('save mesh', time.time()-st)


    if vis_param.n_left_steps == 0:
        return False
    if vis_param.sequence.frame_id >= len(vis_param.sequence):
        # finished, not save model
        print('saving to', vis_param.args.outdir+'/nslfs.pt')
        vis_param.color_map.save_nslfs(vis_param.args.outdir+'/nslfs.pt')
        
        raise StopIteration
        


    vis_param.n_left_steps -= 1

    logging.info(f"Frame ID = {vis_param.sequence.frame_id}")
    frame_data = next(vis_param.sequence)

    if (vis_param.sequence.frame_id - 1) % vis_param.args.color_integrate_interval != 0:
        return False

    # Prune invalid depths
    frame_data.depth[torch.logical_or(frame_data.depth < vis_param.args.depth_cut_min,
                                      frame_data.depth > vis_param.args.depth_cut_max)] = np.nan
    # Do tracking.
    st = time.time()
    if vis_param.sequence.gt_trajectory is not None:
        frame_pose = vis_param.tracker.track_camera(frame_data.rgb, frame_data.depth, frame_data.calib, vis_param.sequence.gt_trajectory[vis_param.sequence.frame_id-1])
    else:
        frame_pose = vis_param.tracker.track_camera(frame_data.rgb, frame_data.depth, frame_data.calib, vis_param.sequence.first_iso if len(vis_param.tracker.all_pd_pose) == 0 else None)

    print('track time:', time.time()-st, "now takes", time.time()-it_st)

    calib = frame_data.calib
    tracker_pc, tracker_normal = vis_param.tracker.last_processed_pc_color_use
    _, tracker_color = vis_param.tracker.last_colored_pcd

    info = (frame_pose, [calib.fx,calib.fy,calib.cx,calib.cy],tracker_color, frame_data.depth)
    if (vis_param.sequence.frame_id - 1) % vis_param.args.color_integrate_interval == 0:
    # integrate into color 
        opt_depth = frame_pose @ tracker_pc
        
        opt_normal = None#opt_normal = frame_pose.rotation @ tracker_normal
        vis_param.color_map.integrate_keyframe(opt_depth, opt_normal, async_optimize=vis_param.args.run_async,
                                         do_optimize=False, info=info)
        vis_param.whole_pc = np.concatenate([vis_param.whole_pc,opt_depth.cpu().numpy()],axis=0)
    tracker_pc, tracker_normal = vis_param.tracker.last_processed_pc
    # integrate into shape   
    if frame_pose is None: # bad frame in track
        return True
    if (vis_param.sequence.frame_id - 1) % vis_param.args.integrate_interval == 0:
        st = time.time()

        opt_depth = frame_pose @ tracker_pc
        opt_normal = frame_pose.rotation @ tracker_normal
        vis_param.map.integrate_keyframe(opt_depth, opt_normal, async_optimize=vis_param.args.run_async,
                                         do_optimize=True)
        print('integrate difusion', time.time()-st, "now takes", time.time()-it_st)
    if (vis_param.sequence.frame_id - 1) % vis_param.args.meshing_interval == 0:
        # extract mesh
        torch.cuda.empty_cache()
        map_mesh = vis_param.map.extract_mesh(vis_param.args.resolution, int(4e6), max_std=0.15,
                                                  extract_async=vis_param.args.run_async, interpolate=True)
        print('extract mesh', time.time()-st)
        vis_param.mesh_updated = map_mesh is not None

    prev_frame = frame_data
    print('################### This iter takes %f seconds'%(time.time()-it_st))
    return True


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
def vr():
    global map_mesh_copy, map_mesh_updated

    pygame.init()
    clock = pygame.time.Clock()


    # pose
    first_iso = vis_param.sequence.first_iso#Isometry(q=Quaternion(array=first_tq[3:]), t=np.array(first_tq[:3]))
    change_pose = first_iso.matrix

    heading_bias = change_pose[:3,:3]#.dot(np.array([[1.,0,0]]).T)
    heading = np.eye(3)
    posi = change_pose[:3,(3,)]



    # raycaster
    H, W = 240, 320
    calib_matrix = np.eye(3)
    calib_matrix[0,0] = 320
    calib_matrix[1,1] = 240
    calib_matrix[0,2] = 160
    calib_matrix[1,2] = 120



    while map_mesh_copy is None:
        time.sleep(1.)

    #vis_param.ray_caster = RayCaster(copy(map_mesh), H, W, calib_matrix)

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

        st = time.time()
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

        print('1. key:', time.time()-st)
        # 2. predict
        if map_mesh_updated:
            vis_param.ray_caster = RayCaster(map_mesh_copy, H, W, calib_matrix)
            map_mesh_updated = False
        pts = vis_param.ray_caster.ray_cast(pose) # N,3
        print('2. ray cast:', time.time()-st)
        invalid_mask = np.isinf(pts)
        pts[invalid_mask] = 0.
        pts[np.isnan(pts)] = 0.
           
        info = (pose, None, None, None)
        color = vis_param.color_map.integrate_keyframe(torch.tensor(pts).cuda().float(), None,info=info, vr_along_train=True) # N,3
        color[invalid_mask] = 0.
        im = color.reshape((vis_param.ray_caster.H,vis_param.ray_caster.W,3)).cpu().detach().numpy()
        print('3. color predict:', time.time()-st)

        del color
        torch.cuda.empty_cache() 

        # 3. display
        image = im.transpose(1,0,2)*255
        surf = pygame.surfarray.make_surface(image)
        display.blit(surf, (0, 0))
        pygame.display.update()







if __name__ == '__main__':
    parser = exp_util.ArgumentParserX()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # Load in network.  (args.model is the network specification)
    model, args_model = utility.load_model(args.training_hypers, args.using_epoch)
    args.model = args_model
    args.mapping = exp_util.dict_to_args(args.mapping)
    args.color_mapping = exp_util.dict_to_args(args.color_mapping)
    args.tracking = exp_util.dict_to_args(args.tracking)

    # Load in sequence.
    seq_package, seq_class = args.sequence_type.split(".")
    sequence_module = importlib.import_module("dataset.production." + seq_package)
    sequence_module = getattr(sequence_module, seq_class)
    vis_param.sequence = sequence_module(**args.sequence_kwargs)

    # Mapping
    if torch.cuda.device_count() > 1:
        main_device, aux_device = torch.device("cuda", index=0), torch.device("cuda", index=1)
    elif torch.cuda.device_count() == 1:
        main_device, aux_device = torch.device("cuda", index=0), None
    else:
        assert False, "You must have one GPU."

    vis_param.color_map = color_map.DenseIndexedMap(None, args.color_mapping, 10, main_device,
                                        args.run_async, aux_device)
    vis_param.map = surface_map.DenseIndexedMap(model, args.mapping,  args.model.code_length, main_device,
                                        args.run_async, aux_device)
    vis_param.tracker = system.tracker_surface.SDFTracker(vis_param.map, args.tracking)
    vis_param.args = args
    os.makedirs(args.outdir,  exist_ok = True)


    key_continue(None)
    vr_thread = Thread(target=vr)
    vr_thread.start()
    try:
        while True:
            refresh(None)
    except StopIteration:
        pygame.quit()
        vr_thread.join()
        assert False, "stopped"
        #os._exit(0)
        
