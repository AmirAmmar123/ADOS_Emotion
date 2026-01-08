# #!/usr/bin/python

import numpy as np

# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector
def rigid_transform_3D(A, B):
    """find Transformation from A to B, to tranform: B = (R@A)+t\n
       A and B should be 3xN"""
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < 0, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

# import numpy as np
# from scipy.linalg import orthogonal_procrustes

# def compute_transformation_matrices(point_cloud1, point_cloud2):
#     # Check if the point clouds have the same number of points
#     if point_cloud1.shape[0] != point_cloud2.shape[0]:
#         raise ValueError("The two point clouds must have the same number of points.")

#     # Perform Procrustes analysis to find the optimal transformation
#     rotation, translation = orthogonal_procrustes(point_cloud1, point_cloud2)

#     # Compute the scaling matrix
#     scaling = np.sqrt(np.sum(point_cloud2 ** 2, axis=0) / np.sum(point_cloud1 ** 2, axis=0))
#     scaling_matrix = np.diag(scaling)

#     return rotation, translation, scaling_matrix

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes

def compute_transformation_matrices(point_cloud1, point_cloud2):
    # Check if the point clouds have the same number of points
    if point_cloud1.shape[0] != point_cloud2.shape[0]:
        raise ValueError("The two point clouds must have the same number of points.")

    # Perform Procrustes analysis to find the optimal transformation
    rotation,translation = rigid_transform_3D(point_cloud1, point_cloud2)

    # Compute the scaling matrix
    scaling = np.sqrt(np.sum(point_cloud1 ** 2, axis=0) / np.sum(point_cloud2 ** 2, axis=0))
    scaling_matrix = np.diag(scaling)

    return rotation, translation, scaling_matrix

# Test the function
# from scipy.spatial.transform import Rotation as R


# # Generate random rotation angles for each axis
# theta_x = np.random.uniform(0, 2*np.pi)
# theta_y = np.random.uniform(0, 2*np.pi)
# theta_z = np.random.uniform(0, 2*np.pi)

# # Calculate the rotation matrix
# rotation_matrix = np.array([
#     [np.cos(theta_y)*np.cos(theta_z), -np.cos(theta_x)*np.sin(theta_z) + np.sin(theta_x)*np.sin(theta_y)*np.cos(theta_z), np.sin(theta_x)*np.sin(theta_z) + np.cos(theta_x)*np.sin(theta_y)*np.cos(theta_z)],
#     [np.cos(theta_y)*np.sin(theta_z), np.cos(theta_x)*np.cos(theta_z) + np.sin(theta_x)*np.sin(theta_y)*np.sin(theta_z), -np.sin(theta_x)*np.cos(theta_z) + np.cos(theta_x)*np.sin(theta_y)*np.sin(theta_z)],
#     [-np.sin(theta_y), np.sin(theta_x)*np.cos(theta_y), np.cos(theta_x)*np.cos(theta_y)]
# ])

# print(rotation_matrix)

# point_cloud1 = np.random.rand(4,3)
# print(point_cloud1.shape)
# # print(randomR)
# point_cloud2 = (point_cloud1@rotation_matrix)
# tx = 2
# ty = 10
# tz = 100

# # Construct the translation matrix
# translation_matrix = [
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, 1, 0],
#     [tx, ty, tz, 1],
# ]

# # Apply the transformation to our point cloud
# translated_points = np.matmul(
#     point_cloud_homogeneous,
#     translation_matrix)

# # Convert to cartesian coordinates
# translated_points_xyz = []
# for point in translated_points:

#     point = np.array(point[:-1])
#     translated_points_xyz.append(point)print(point_cloud2.shape)
# newR,t = rigid_transform_3D(point_cloud2.T,point_cloud1.T)
# print(t)
# print(point_cloud2 - ((point_cloud1@newR)+t.T))
# print(rotation_matrix-newR)
# print(point_cloud2.shape)
# rotation, translation, scaling_matrix = compute_transformation_matrices(point_cloud1, point_cloud2)
# print(rotation-randomR)
# Transform point cloud 2 to align with point cloud 1
# transformed_point_cloud2 = (point_cloud2 @ scaling_matrix) @ rotation.T + translation

# print("Transformed Point Cloud 2:")
# print(transformed_point_cloud2)


def create_binary_dict(size=3):
    options = [True,False]
    ret_dict = {}
    for i in range(2**size):
        num = i
        lst = []
        runs = 0
        while(runs<size):
            runs+=1
            lst.append((num%2)==1)
            num/=2
            num = int(num)
        ret_dict[tuple(lst)]=[]
    return ret_dict

def get_frames_in_range_ms(start,end,df):
    start = start/1000
    end = end/1000
    min_set_flag = False
    min_f = float('inf')
    max_f = -1
    for index,row in df.iterrows():
        if float(row['timestamp']) > end:
            break
        if float(row['timestamp']) < start:
            continue
        if not min_set_flag:
            min_f = index
            min_set_flag = True
        max_f = index
    return (min_f,max_f)
        
