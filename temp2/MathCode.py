import numpy as np
def intersect_vector_plane(v, p1, p2, p3):
    # v is the vector represented by a 3D point and direction
    # p1, p2, p3 are the 3 points that define the plane
    v0, d = v

    # calculate the normal vector of the plane
    u = [p2[i] - p1[i] for i in range(3)]
    v = [p3[i] - p1[i] for i in range(3)]
    n = [u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2], u[0]*v[1] - u[1]*v[0]]

    # calculate the intersection point
    t = sum(n[i]*(p1[i] - v0[i]) for i in range(3)) / sum(n[i]*d[i] for i in range(3))
    intersection = [v0[i] + t*d[i] for i in range(3)]

    # return None if the intersection is not on the line

    # return the intersection point as a list
    return intersection
def intersect_vector_plane(v, p):
    # v is the vector represented by a 3D point and direction
    # p1, p2, p3 are the 3 points that define the plane
    # print(p)
    vector_point,vector_direction = v
    
    # Plane representation

    point1, point2, point3 = p
    
    # Calculate the normal vector of the plane
    normal = np.cross(point2 - point1, point3 - point1)
    
    # print(normal)
    # Calculate the distance from the vector point to the plane
    distance = np.dot(point1 - vector_point, normal) / np.dot(vector_direction, normal)
    
    # Calculate the intersection point
    intersection_point = vector_point + distance * vector_direction
    

    # print(intersection_point)
    # return None if the intersection is not on the line

    # return the intersection point as a list
    return intersection_point


def find_x_value_point(vector_point, vector_direction, n):
    # Convert inputs to numpy arrays
    vector_point = np.array(vector_point)
    vector_direction = np.array(vector_direction)
    
    # Calculate the parameter t for the x-value equation
    t = (n - vector_point[0]) / vector_direction[0]
    
    # Calculate the point on the vector where x = n
    x_value_point = vector_point + t * vector_direction
    
    return x_value_point.tolist()


def scale_point_clouds(point_cloud1, point_cloud2):
    # Step 1: Calculate the centroids
    centroid1 = np.mean(point_cloud1, axis=0)
    centroid2 = np.mean(point_cloud2, axis=0)
    
    # Step 2: Translate the point clouds to have centroids at the origin
    translated_cloud1 = point_cloud1 - centroid1
    translated_cloud2 = point_cloud2 - centroid2
    
    # Step 3: Calculate the average distance from the origin for each point cloud
    avg_distance1 = np.mean(np.linalg.norm(translated_cloud1, axis=1))
    avg_distance2 = np.mean(np.linalg.norm(translated_cloud2, axis=1))
    
    # Step 4: Scale the point cloud 1 to match the size of point cloud 2
    scaled_cloud1 = translated_cloud1 * (avg_distance2 / avg_distance1)
    
    return scaled_cloud1, translated_cloud2


def point_line_distance(point,line):

    line_point,line_direction = line
    line_direction = line_direction / np.linalg.norm(line_direction)  # Normalize the line direction vector

    point_to_line = point - line_point
    distance_along_line = np.dot(point_to_line, line_direction)
    
    closest_point = line_point + distance_along_line * line_direction
    
    return closest_point


#
# def rotate_vector_around_axis(vector, axis, angle):
#     axis = axis / np.linalg.norm(axis)
#     a = np.cos(angle / 2.0)
#     b, c, d = -axis * np.sin(angle / 2.0)
#     rotation_matrix = np.array([
#         [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
#         [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
#         [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]
#     ])
#     return np.dot(rotation_matrix, vector)

# def angle_between_vectors(v1,v2):
#     return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
def angle_between_vectors(v1, v2):
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def rotate_vector_around_axis(vector, axis, angle):
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    rotation_matrix = np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]
    ])
    return np.dot(rotation_matrix, vector)
def find_vector_with_same_dot_product(dot_product_v1_v2, v3):
    v4 = dot_product_v1_v2 / v3
    # v4 = (dot_product_v1_v2 / np.dot(v3, v3)) * v3
    # v4_magnitude = np.abs(dot_product_v1_v2 / np.dot(v3, v3))
    # v4 = v4_magnitude * v3
    return v4

def find_complement_vector_dot(v1,v2,v3):
    """find v4 vector, that has the same dot product with v3 as v1dotv2"""
    # v1 = v1 / np.linalg.norm(v1)
    # v2 = v2 / np.linalg.norm(v2)
    dot_product_v1_v2 =  np.dot(v1,v2)
    return find_vector_with_same_dot_product(dot_product_v1_v2,v3)
def find_complement_vector_angle(v1,v2,v3):
    """find v4 vector, that has the same dot product with v3 as v1dotv2"""
    # v1 = v1 / np.linalg.norm(v1)
    # v2 = v2 / np.linalg.norm(v2)
    angle = angle_between_vectors(v1, v2)
    v4 = rotate_vector_around_axis(v3, np.cross(v1, v2), angle)
    return v4
def find_complement_vector_simple(v1,v2,v3):
    """find v4 vector, that has the same dot product with v3 as v1dotv2"""
    v1 = normalize(v1)
    v2 = normalize(v2)
    v3 = normalize(v3)
    v4 = v3 - (v2-v1)
    v4 = normalize(v4)
    # print(v4)
    # input()
    return v4
def find_complement_vector_complex(v1,v2,v3):
    from numpy import arctan
    a1 = arctan(v1[0])
    b1 = arctan(v1[1])
    c1 = arctan(v1[2])
    a2 = arctan(v2[0])
    b2 = arctan(v2[1])
    c2 = arctan(v2[2])
    a3 = arctan(v3[0])
    b3 = arctan(v3[1])
    c3 = arctan(v3[2])
    a4 = a3+a2-a1
    b4 = b3+b2-b1
    c4 = c3+c2-c1
    return np.array([a4,b4,c4])
def project_point_onto_plane(point, plane_points):
    # Convert input to numpy arrays for easier calculations
    point = np.array(point)
    plane_points = np.array(plane_points)
    
    # Calculate the normal vector of the plane
    v1 = plane_points[1] - plane_points[0]
    v1 = v1 / np.linalg.norm(v1)
    v2 = plane_points[2] - plane_points[0]
    v2 = v2 / np.linalg.norm(v2)
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)
    # new_v2 = -np.cross(v1,normal)
    # Calculate the projection of the point onto the plane
    projection = point - np.dot(point - plane_points[0], normal) * normal
    
    return projection
def get_plane_rot_matrix(point , plane_points):
    # Convert input to numpy arrays for easier calculations
    # point = np.array(point)
    plane_points = np.array(plane_points)
    
    # Calculate the normal vector of the plane
    v1 = plane_points[1] - plane_points[0]
    v1 = v1 / np.linalg.norm(v1)
    v2 = plane_points[2] - plane_points[0]
    v2 = v2 / np.linalg.norm(v2)
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)
    new_v2 = -np.cross(v1,normal)
    # Calculate the projection of the point onto the plane
    rot_mat = np.array([v1,new_v2,normal])   
    #check rot mat:
    # print(rot_mat@v1)
    # print(rot_mat@new_v2)
    # print(rot_mat@normal) 
    return rot_mat
def project_point_onto_plane_v2(point, plane_point,plane_normal):
    # Convert input to numpy arrays for easier calculations
    point = np.array(point)
    plane_point = np.array(plane_point)
    plane_normal = np.array(plane_normal)
    
    # Calculate the projection of the point onto the plane
    projection = point - np.dot(point - plane_point, plane_normal) * plane_normal
    
    return projection
def normalize(v : np.array):
    v = v / np.linalg.norm(v)
    return v.copy()
def find_vector_with_same_angle(v1, v2, v3):
    # Convert vectors to numpy arrays for easier calculations
    v1 = np.array(v1)
    v2 = np.array(v2)
    v3 = np.array(v3)
    
    # Calculate the angle between v1 and v2
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    # Calculate the rotation axis using cross product
    axis = np.cross(v3, v1)
    axis /= np.linalg.norm(axis)
    
    # Create a rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle) + axis[0]**2 * (1 - np.cos(angle)), axis[0] * axis[1] * (1 - np.cos(angle)) - axis[2] * np.sin(angle), axis[0] * axis[2] * (1 - np.cos(angle)) + axis[1] * np.sin(angle)],
        [axis[1] * axis[0] * (1 - np.cos(angle)) + axis[2] * np.sin(angle), np.cos(angle) + axis[1]**2 * (1 - np.cos(angle)), axis[1] * axis[2] * (1 - np.cos(angle)) - axis[0] * np.sin(angle)],
        [axis[2] * axis[0] * (1 - np.cos(angle)) - axis[1] * np.sin(angle), axis[2] * axis[1] * (1 - np.cos(angle)) + axis[0] * np.sin(angle), np.cos(angle) + axis[2]**2 * (1 - np.cos(angle))]
    ])
    
    # Rotate v3 to obtain v4
    v4 = np.dot(rotation_matrix, v3)
    
    return v4