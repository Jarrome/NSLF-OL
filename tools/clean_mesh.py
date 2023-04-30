import sys
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree




ps = np.load('tmp.npy')
mesh = o3d.io.read_triangle_mesh(sys.argv[1])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(ps)

downpcd = pcd.voxel_down_sample(voxel_size=0.01)
p = np.asarray(downpcd.points)
kdtree = KDTree(p)

q = np.asarray(mesh.vertices)
dd, ii = kdtree.query(q)
bad_mask = dd>0.01
q[bad_mask,:] = p[ii[bad_mask],:] 

mesh.vertices = o3d.utility.Vector3dVector(q)



o3d.io.write_triangle_mesh('tmp.ply', mesh)

