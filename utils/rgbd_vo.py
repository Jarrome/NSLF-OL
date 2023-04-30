'''
    taken from https://github.com/ParadoxRobotics/RGBD_VO/blob/master/RGBDVO.py
'''


import numpy as np
from sklearn.neighbors import NearestNeighbors
import cv2
import utils.icp as icp 
# Camera intrinsic parameters
'''
fx = 641.66
fy = 641.66
cx = 324.87
cy = 237.38
'''
fx, fy, cx, cy = 615.3265991210938, 615.4910888671875, 320.8399963378906, 246.25180053710938

depth_scale = 0.0010000000474974513

CIP = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

Rot_pose = np.eye(3)
Tr_pose = np.array([[0],[0],[0]])

# Initiate ORB object
orb = cv2.ORB_create(nfeatures=1000, nlevels=0, scoreType=cv2.ORB_FAST_SCORE)
# Init feature matcher
matcher = cv2.DescriptorMatcher_create("BruteForce-L1")

ref_frame, ref_depth, kp_ref, d_ref = None, None, None, None
T_global = None
def vo_update(cur_frame, cur_depth):
    global ref_frame, ref_depth, kp_ref, d_ref, T_global

    cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    # find keypoints
    kp_cur, d_cur = orb.detectAndCompute(cur_frame, None)


    if ref_frame is not None:
        # make match
        matches = matcher.knnMatch(d_ref, d_cur, 2)

        # filter match using lowe loss
        good_match = []
        good_match_print = [] # only fo debug
        for m,n in matches:
            if m.distance < 0.45*n.distance:
                good_match.append(m)
                good_match_print.append([m])
        # print matches
        #img3 = cv2.drawMatchesKnn(ref_frame, kp_ref, cur_frame, kp_cur, good_match_print, None, flags=2)
        #cv2.imshow('state', img3)

        # create 2 points clouds with 5 random points pc = [x,y,z]
        ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_match])
        cur_pts = np.float32([kp_cur[m.trainIdx].pt for m in good_match])
        pc_ref = np.zeros((len(ref_pts),3))
        pc_cur = np.zeros((len(cur_pts),3))

        for id in range(len(ref_pts)):
            # z component
            pc_ref[id, 2] = ref_depth[int(ref_pts[id, 1]), int(ref_pts[id, 0])]*depth_scale
            pc_cur[id, 2] = cur_depth[int(cur_pts[id, 1]), int(cur_pts[id, 0])]*depth_scale
            # x component
            pc_ref[id, 0] = (ref_pts[id, 0]-cx)*(pc_ref[id, 2]/fx)
            pc_cur[id, 0] = (cur_pts[id, 0]-cx)*(pc_cur[id, 2]/fx)
            # y component
            pc_ref[id, 1] = (ref_pts[id, 1]-cy)*(pc_ref[id, 2]/fy)
            pc_cur[id, 1] = (cur_pts[id, 1]-cy)*(pc_cur[id, 2]/fy)

        # ICP
        T, distances, iterations = icp.icp(pc_cur, pc_ref, tolerance=0.000001)
        T_global = T.dot(T_global)
        #ROT = T[0:3, 0:3]
        #TR = np.array([[T[0,3]],[T[1,3]],[T[2,3]]])
    else:
        T_global = np.eye(4)


    # update
    ref_frame = cur_frame
    ref_depth = cur_depth
    kp_ref = kp_cur
    d_ref = d_cur

    return T_global
