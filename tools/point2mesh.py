import numpy as np
import open3d as o3d



ps = np.load('tmp.npy')
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(ps)

downpcd = pcd.voxel_down_sample(voxel_size=0.01)
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05,
                                                          max_nn=30))
radii = [0.02, 0.04]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    downpcd, o3d.utility.DoubleVector(radii))

'''
print('run Poisson surface reconstruction')
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        downpcd, depth=9)
'''
o3d.io.write_triangle_mesh('tmp.ply', mesh)

