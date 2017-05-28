import numpy as np
from image_utils import array_to_image

# degree to radians
D2R = (np.pi / 180)

V_RES_DEG = 1.0  # 0.9
H_RES_DEG = 0.9  # 75 #0.3
V_FOV_DEG = (-30.67, 10.67)
V_FOV_TOTAL_DEG = -V_FOV_DEG[0] + V_FOV_DEG[1]
Y_FUDGE = 0.0

# CONVERT TO RADIANS
V_RES_RAD = V_RES_DEG * D2R
H_RES_RAD = H_RES_DEG * D2R
V_FOV_RAD = (V_FOV_DEG[0] * D2R, V_FOV_DEG[1] * D2R)
V_FOV_TOTAL_RAD = V_FOV_TOTAL_DEG * D2R

D_PLANE = (V_FOV_TOTAL_DEG / V_RES_DEG) / V_FOV_TOTAL_RAD
H_AVOVE_AND_BELOW = D_PLANE * (np.tan(-V_FOV_RAD[0]) + np.tan(V_FOV_RAD[1]))

# THEORETICAL LIMITS FOR IMAGE
Y_MIN = -((V_FOV_DEG[1] / V_RES_DEG) + Y_FUDGE)
Y_MAX = int(np.ceil(H_AVOVE_AND_BELOW + Y_FUDGE))
X_MIN = -180.0 / H_RES_DEG
X_MAX = int(np.ceil(360.0 / H_RES_DEG))
IMG_DIMENSIONS = [Y_MAX + 1, X_MAX + 1]


def distance(x, y, z=0.0):
    return np.sqrt(np.add(np.add(x ** 2, y ** 2), z ** 2))


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotate_around_center(x, y, z, theta):
    rm = rotation_matrix([0.0, 0.0, 1.0], theta)
    return np.nan_to_num(np.dot(np.column_stack((x, y, z)), rm.T))


def unwrap_point(x, y, z, d, rotation_deg=0.0):
    theta = rotation_deg * float(D2R)
    if theta != 0:
        x, y, z = rotate_around_center(x, y, z, theta).T

    # MAP TO CYLINDER
    x_img = np.arctan2(y, x) / H_RES_RAD
    y_img = -(np.arctan2(z, d) / V_RES_RAD)

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_img = np.trunc(-x_img - X_MIN).astype(np.int32)
    y_img = np.trunc(y_img - Y_MIN).astype(np.int32)

    return x_img, y_img


def point_cloud_to_panorama(points,
                            d_range=(-10.0, 200.0),
                            rotation_deg=0.0,
                            ):
    # map distance relative to origin
    # z is up, x is forward, y is left
    points['map_distance'] = distance(points['x'], points['y'])  # np.sqrt(points['x']**2 + points['y']**2)
    # print("map_distance", points['map_distance'].shape)
    # print("map_distance min: %s    max:  %s"%(points['map_distance'].min(), points['map_distance'].max()))

    # Filter by distance
    points = points[points['map_distance'] > d_range[0]]
    points = points[points['map_distance'] < d_range[1]]
    # print("filtered points", points.shape)

    d_points = points['map_distance']
    r_points = np.sqrt(points['intensity'])

    x_points, y_points = unwrap_point(points['x'], points['y'], points['z'], d_points, rotation_deg=rotation_deg)

    xy_points = [x_points, y_points]

    img_distance = array_to_image(d_points, xy_points, IMG_DIMENSIONS)
    img_intensity = array_to_image(r_points, xy_points, IMG_DIMENSIONS)

    return img_distance, img_intensity
