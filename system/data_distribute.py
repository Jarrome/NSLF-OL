import sys
import os
import torch
from torch import nn, optim
import numpy as np
import cv2
import open3d as o3d
from time import time, sleep
import pdb
import matplotlib.pyplot as plt
from system.color_thread import background
import asyncio 

@background
def feed_data(data):
    imp, voxel_ids, inverse_indices, id, d_xyz, ds_flat, rgb, is_train = data
    if imp.indexer[voxel_ids[id]] != -1:
        mask = inverse_indices == id

        if is_train:
            imp.nslfs[voxel_ids[id].item()].add_data(d_xyz[mask,:], ds_flat[mask,:],rgb[mask,:],is_train)
        else:
            imp.nslfs[voxel_ids[id].item()].add_data(d_xyz[mask,:], ds_flat[mask,:],None,is_train)

def infer_data(data):
    imp, voxel_ids, inverse_indices, id, d_xyz, ds_flat, rgb, is_train = data
    if imp.indexer[voxel_ids[id]] != -1:
        mask = inverse_indices == id

        if is_train:
            assert False, 'only test'
        else:
            pred = imp.nslfs[voxel_ids[id].item()].eval_w_input(d_xyz[mask,:], ds_flat[mask,:])
        return pred


@background
def feed_iter(data):
    imp, key, iter_nm = data
    imp.nslfs[key].set_iters(iter_nm)


def get_color_nonthread(imp, xyzn, rgb, device, opt=True, iter_nm=5):
    st = time()
    r_ts_flat,ds_flat = xyzn

    surface_xyz_zeroed = r_ts_flat - imp.bound_min.unsqueeze(0)
    surface_xyz_normalized = surface_xyz_zeroed / imp.voxel_size
    vertex = torch.ceil(surface_xyz_normalized).long() - 1
    surface_grid_id = imp._linearize_id(vertex)
    surface_grid_id[surface_grid_id > imp.indexer.shape[0]] = -1
    invalid_surface_ind = torch.logical_or(imp.indexer[surface_grid_id] == -1 , (surface_grid_id < 0))
    #np.save('tmp1.npy', r_ts_flat[~invalid_surface_ind,:].cpu())
    d_xyz = surface_xyz_normalized - vertex - 0.5


    voxel_ids, inverse_indices = torch.unique(surface_grid_id, return_inverse=True)



    c_is = torch.zeros(r_ts_flat.shape).to(device)
    sigma_is = torch.zeros(r_ts_flat.shape[0]).to(device)

    for id in range(voxel_ids.shape[0]):
        if imp.indexer[voxel_ids[id]] == -1:
            continue
        mask = inverse_indices == id
        if voxel_ids[id].item() < 0: 
                continue

      
        pred = infer_data((imp, voxel_ids, inverse_indices, id, d_xyz, ds_flat, rgb, opt)).detach().float()
        c_is[mask,:] = pred
    return c_is  



def get_color(imp, xyzn, rgb, device, opt=True, iter_nm=5):
    st = time()
    r_ts_flat,ds_flat = xyzn

    surface_xyz_zeroed = r_ts_flat - imp.bound_min.unsqueeze(0)
    surface_xyz_normalized = surface_xyz_zeroed / imp.voxel_size
    vertex = torch.ceil(surface_xyz_normalized).long() - 1
    surface_grid_id = imp._linearize_id(vertex)
    surface_grid_id[surface_grid_id > imp.indexer.shape[0]] = -1
    invalid_surface_ind = torch.logical_or(imp.indexer[surface_grid_id] == -1 , (surface_grid_id < 0))
    #np.save('tmp1.npy', r_ts_flat[~invalid_surface_ind,:].cpu())
    d_xyz = surface_xyz_normalized - vertex - 0.5


    voxel_ids, inverse_indices = torch.unique(surface_grid_id, return_inverse=True)


    #print('c', time()-st)
    for _ in range(iter_nm):
        params = []

        #print("region id:",voxel_ids)
        for id in range(voxel_ids.shape[0]):
            feed_data((imp, voxel_ids, inverse_indices, id, d_xyz, ds_flat, rgb, opt))
            '''
            if imp.indexer[voxel_ids[id]] == -1:
                continue
            mask = inverse_indices == id
            imp.nslfs[voxel_ids[id].item()].add_data(d_xyz[mask,:], ds_flat[mask,:],rgb[mask,:])
            '''
        #print('cd', time()-st)

        if rgb is not None:
            # always add iters for the few trained voxels
            counts = [imp.nslfs[nf_key].trained_iters for nf_key in imp.nslfs.keys()]

            #print(counts)
            #print(imp.nslfs.keys())
            #idx = np.argsort(counts)
            keys = list(imp.nslfs.keys())
            idx = np.arange(len(counts))
            print("keys",keys)
            print("counts",counts)
            mean = np.mean(counts) + 20#1000 #50 # 10 for MACA, 1000 for one_HG

            if False:#mean < 100:
                maintained_CA_nm = int(len(counts))
                need_iters = dict()
                for id in idx:
                    need_iters[keys[id]] = 100 if counts[id] < 100 else 0
            else:
                #maintained_CA_nm = 4

                need_iters = dict()
                for id in idx:
                    need_iters[keys[id]] = max(mean- counts[id], 0)


            #good_keys = [keys[id] for id in idx[:idx.shape[0]//2]]
            #bad_keys = [keys[id] for id in idx[idx.shape[0]//2:]]
            #good_keys = [keys[id] for id in idx[:maintained_CA_nm]]
            #bad_keys = [keys[id] for id in idx[maintained_CA_nm:]]
            print("Median", mean)
            print(need_iters.keys())
            print(need_iters.values())
            for key in keys:
                feed_iter((imp, key, need_iters[key]))

            '''
            #print('keys',keys)
            print(good_keys)
            for key in good_keys:#imp.nslfs.keys():
                #imp.nslfs[key].add_iters(10)
                feed_iter((imp, key, 100)) # 100 works well on icl and replica
    #print('d', time()-st)
            print(bad_keys)
            for key in bad_keys:
                feed_iter((imp, key, 0))
            '''




    if not opt: # eval mode, collect the prediction
        
        c_is = torch.zeros(r_ts_flat.shape).to(device)
        sigma_is = torch.zeros(r_ts_flat.shape[0]).to(device)


        for id in range(voxel_ids.shape[0]):
            if imp.indexer[voxel_ids[id]] == -1:
                continue
            #print('collect',voxel_ids[id].item())
            mask = inverse_indices == id
            if voxel_ids[id].item() < 0: 
                continue
            while imp.nslfs[voxel_ids[id].item()].pred is None:
                sleep(.01)
            pred = imp.nslfs[voxel_ids[id].item()].pred.detach().float()
            imp.nslfs[voxel_ids[id].item()].pred = None
            c_is[mask,:] = pred
        return c_is  
    #print('e', time()-st)


def train_once(imp, pose, calib, rgb, depth_im, xyz, device):
    #torch.backends.cudnn.benchmark = True
    fx,fy,tx,ty = calib

    ds = xyz-torch.from_numpy(pose[:3,(3,)].T).to(xyz)
    ds /= torch.norm(ds+1e-8,dim=1,keepdim=True)

    #eval_once(imp, pose, calib, depth_im, device)

    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(rgb.cpu().numpy())
    o3d.io.write_point_cloud('tmp.ply', pcd)
    '''
    get_color(imp, (xyz, ds), rgb, device, opt=True, iter_nm=1)#50)

    #color = get_color(imp, (xyz, ds), rgb, device, opt=False, iter_nm=5)

    #eval_once(imp, pose, calib, depth_im, device)

from system.ext import unproject_depth
import cv2

def eval_once(imp, pose_, calib, depth_im, device):
    pose = torch.from_numpy(pose_).to(depth_im)

    fx,fy,tx,ty = calib
    pc_data = unproject_depth(depth_im, fx, fy, tx, ty)
    xyz = pc_data.reshape((-1,3))
    xyz = (pose[:3,:3]@xyz.T+pose[:3,(3,)]).T

    ds = xyz-pose[:3,(3,)].T
    ds /= torch.norm(ds,dim=1,keepdim=True)

    color = get_color(imp, (xyz, ds), None, device, opt=False, iter_nm=1)

    color = color.reshape((pc_data.shape)).cpu().detach().numpy()

    cv2.imshow("image", color)
    cv2.waitKey(1)

def vis_once(imp, pose, xyz, device):
    ds = xyz-torch.from_numpy(pose[:3,(3,)].T).to(xyz)
    ds /= torch.norm(ds+1e-8,dim=1,keepdim=True)
    color = get_color(imp, (xyz, ds), None, device, opt=False, iter_nm=1)

    # for AFFM, the async inf run out of memory
    #color = get_color_nonthread(imp, (xyz, ds), None, device, opt=False, iter_nm=1)

    return color


'''
def vis_once(imp, pose, xyz, device):
    ds = xyz-torch.from_numpy(pose[:3,(3,)].T).to(xyz)
    ds /= torch.norm(ds+1e-8,dim=1,keepdim=True)
    color = get_color(imp, (xyz, ds), None, device, opt=False, iter_nm=1)

    return color
'''




