#!/usr/bin/env python
import sys
import os
import numpy as np
import cv2
import pcl
import glob
import math
import pickle
import parse_xml as pt


def load_pc_from_pcd(pcd_path):
    """Load PointCloud data from pcd file."""
    p = pcl.load(pcd_path)
    return np.array(list(p), dtype=np.float32)


def read_label_from_xml(label_path):
    """Read label from xml file.

    # Returns:
        labe    l_dic (dictionary): labels for one sequence.
        size (list): Bounding Box Size. [l, w. h]?
    """
    labels = pt.parse_xml(label_path)
    label_dic = {}
    for label in labels:
        first_frame = label.first_frame
        num_frames = label.num_frames
        size = label.size
        obj_type = label.object_type
        for index, place, rot in zip(range(first_frame, first_frame + num_frames), label.trans, label.rots):
            if index in label_dic.keys():
                label_dic[index]["trans"] = np.vstack((label_dic[index]["trans"], place))
                label_dic[index]["size"] = np.vstack((label_dic[index]["size"], np.array(size)))
                label_dic[index]["rot"] = np.vstack((label_dic[index]["rot"], rot))
            else:
                label_dic[index] = {}
                label_dic[index]["trans"] = place
                label_dic[index]["rot"] = rot
                label_dic[index]["size"] = np.array(size)
    return label_dic, size


def get_boxcorners(places, rots, size):
    """Create 8 corners of bounding box from bottom center."""
    corners = []
    try:
        zip(places, rots, size)
    except: 
        import pdb
        pdb.set_trace()
        
    for place, rot, sz in zip(places, rots, size):
        x, y, z = place
        h, w, l = sz
        if l > 10:
            continue

        corner = np.array([
            [x - l / 2.0, y - w / 2.0, z],
            [x + l / 2.0, y - w / 2.0, z],
            [x - l / 2.0, y + w / 2.0, z],
            [x - l / 2.0, y - w / 2.0, z + h],
            [x - l / 2.0, y + w / 2.0, z + h],
            [x + l / 2.0, y + w / 2.0, z],
            [x + l / 2.0, y - w / 2.0, z + h],
            [x + l / 2.0, y + w / 2.0, z + h],
        ])

        corner -= np.array([x, y, z])

        rot_matrix = np.array([
            [np.cos(rot), -np.sin(rot), 0],
            [np.sin(rot), np.cos(rot), 0],
            [0, 0, 1]
        ])

        a = np.dot(corner, rot_matrix.transpose())
        a += np.array([x, y, z])
        corners.append(a)
    return np.array(corners)


rospy_node = None
def publish_pc2(pc, obj):
    """Publisher of PointCloud data"""
    global rospy_node
    import rospy
    import sensor_msgs.point_cloud2 as pc2
    from sensor_msgs.msg import PointCloud2
    import std_msgs

    if rospy_node is None:
        rospy_node = rospy.init_node("pc2_publisher")

    pub = rospy.Publisher("/points_raw", PointCloud2, queue_size=1000000)

    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "velodyne"
    points = pc2.create_cloud_xyz32(header, pc[:, :3])

    pub2 = rospy.Publisher("/points_raw1", PointCloud2, queue_size=1000000)
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "velodyne"
    points2 = pc2.create_cloud_xyz32(header, obj)

    #r = rospy.Rate(0.1)
    #while not rospy.is_shutdown():
    pub.publish(points)
    pub2.publish(points2)
    #r.sleep()


def pc2voxel(pc, resolution=0.50, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5)):
    """Convert PointCloud2 to Voxel"""
    logic_x = np.logical_and(pc[:, 0] >= x[0], pc[:, 0] < x[1])
    logic_y = np.logical_and(pc[:, 1] >= y[0], pc[:, 1] < y[1])
    logic_z = np.logical_and(pc[:, 2] >= z[0], pc[:, 2] < z[1])

    pc = pc[:, :3][np.logical_and(logic_x, np.logical_and(logic_y, logic_z))]
    pc = ((pc - np.array([x[0], y[0], z[0]])) / resolution).astype(np.int32)

    voxel = np.zeros((int((x[1] - x[0]) / resolution),
                      int((y[1] - y[0]) / resolution),
                      int(round((z[1] - z[0]) / resolution))
                      ))
    voxel[pc[:, 0], pc[:, 1], pc[:, 2]] = 1
    return voxel


def center2sphere(places, size, resolution=0.5, min_value=np.array([0.0, 0.0, 0.0]), scale=4, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5)):
    """Convert object label to Training label for objectness loss"""
    logic_x = np.logical_and(places[:, 0] >= x[0], places[:, 0] < x[1])
    logic_y = np.logical_and(places[:, 1] >= y[0], places[:, 1] < y[1])
    logic_z = np.logical_and(places[:, 2] >= z[0], places[:, 2] < z[1])

    xyz_logical = np.logical_and(logic_x, np.logical_and(logic_y, logic_z))
    center = places.copy()
    center[:, 2] = center[:, 2] + size[:, 0] / 2.0
    sphere_center = ((center[xyz_logical] - min_value) / (resolution * scale)).astype(np.int32)
    return sphere_center


def sphere2center(p_sphere, resolution=0.5, scale=4, min_value=np.array([0.0, 0.0, 0.0])):
    """from sphere center to label center"""
    center = p_sphere * (resolution * scale) + min_value
    return center

def get_label_path_for_point_path(point_path):
    ppath_parts = point_path.split('/')
    ppath_parts[-1] = ppath_parts[-1].replace('pcd', 'pickle')
    ppath_parts[-2] = 'labels'
    return os.path.join("/", *ppath_parts)

def read_labels(label_path):
    places = None
    rots = None
    size = None

    if label_path.endswith(".xml"):
        bboxes, size = read_label_from_xml(label_path)
        places = [bboxes[0]["trans"]]
        rots = [bboxes[0]["rot"][2]]

    if label_path.endswith(".pickle"):
        with open(label_path, 'rb') as in_file:
            size, place, rot = pickle.load(in_file)
        size = [size]
        places = [place]
        rots = [rot]

    return np.array(places), np.array(rots), np.array(size)


def create_label(places, size, corners, resolution=0.50, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5), scale=4, min_value=None):
    if min_value is None:
        min_value = [0.0, 0.0, 0.0]
    min_value = np.array(min_value)

    places = np.array(places)
    size = np.array(size)
    corners = np.array(corners)

    """Create training Labels"""
    x_logical = np.logical_and((places[:, 0] < x[1]), (places[:, 0] >= x[0]))
    y_logical = np.logical_and((places[:, 1] < y[1]), (places[:, 1] >= y[0]))
    z_logical = np.logical_and((places[:, 2] + size[:, 0] / 2.0 < z[1]), (places[:, 2] + size[:, 0] / 2.0 >= z[0]))
    xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical))

    center = places.copy()
    center[:, 2] = center[:, 2] + size[:, 0] / 2.0  # Move bottom to center
    sphere_center = ((center[xyz_logical] - min_value) / (resolution * scale)).astype(np.int32)

    train_corners = corners[xyz_logical].copy()
    anchor_center = sphere2center(sphere_center,
                                     resolution=resolution,
                                     scale=scale,
                                     min_value=min_value)

    for index, (corner, center) in enumerate(zip(corners[xyz_logical], anchor_center)):
        train_corners[index] = corner - center

    return sphere_center, train_corners


def corner_to_train(corners, sphere_center, resolution=0.50, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5), scale=4, min_value=None):
    if min_value is None:
        min_value = [0.0, 0.0, 0.0]

    min_value = np.array(min_value)
    """Convert corner to Training label for regression loss"""
    x_logical = np.logical_and((corners[:, :, 0] < x[1]), (corners[:, :, 0] >= x[0]))
    y_logical = np.logical_and((corners[:, :, 1] < y[1]), (corners[:, :, 1] >= y[0]))
    z_logical = np.logical_and((corners[:, :, 2] < z[1]), (corners[:, :, 2] >= z[0]))
    xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical)).all(axis=1)

    train_corners = corners[xyz_logical].copy()
    sphere_center = sphere2center(sphere_center,
                                     resolution=resolution,
                                     scale=scale,
                                     min_value=min_value)

    for index, (corner, center) in enumerate(zip(corners[xyz_logical], sphere_center)):
        train_corners[index] = corner - center

    return train_corners


def corner_to_voxel(voxel_shape, corners, sphere_center, scale=4):
    """Create final regression label from corner"""
    # import pdb
    # pdb.set_trace()
    corner_voxel = np.zeros((int(voxel_shape[0] / scale),
                             int(voxel_shape[1] / scale),
                             int(voxel_shape[2] / scale), 24))
    corner_voxel[sphere_center[:, 0], sphere_center[:, 1], sphere_center[:, 2]] = corners
    return corner_voxel


def create_objectness_label(sphere_center, resolution=0.5, x=90, y=100, z=10, scale=4):
    """Create Objectness label"""
    obj_maps = np.zeros((int(x / (resolution * scale)),
                         int(y / (resolution * scale)),
                         int(np.round(z / (resolution * scale)))
                         ))
    obj_maps[sphere_center[:, 0], sphere_center[:, 1], sphere_center[:, 2]] = 1
    return obj_maps


def process(points_path, label_path):
    p = []
    pc = load_pc_from_pcd(points_path)

    places, rots, size = read_labels(label_path)

    corners = get_boxcorners(places, rots, size)

    p.append((0, 0, 0))
    p.append((0, 0, -1))
    print(pc.shape)
    # publish_pc2(pc, obj)
    a = center2sphere(places, size, resolution=0.25)
    print(places)
    print(a)
    print(sphere2center(a, resolution=0.25))
    bbox = sphere2center(a, resolution=0.25)
    print(corners.shape)
    # publish_pc2(pc, bbox.reshape(-1, 3))
    publish_pc2(pc, corners.reshape(-1, 3))


if __name__ == "__main__":
    pcd_path = "points/*.pcd"
    label_path = "labels/*.pickle"
    process(pcd_path, label_path)
