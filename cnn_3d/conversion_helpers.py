import sys
import os
import numpy as np
import pandas as pd
import cv2
import pcl
import glob
import math
import errno
import pickle
import six
import parse_xml as pt
import h5py
import input_helpers as ih


def mkdir_p(path):
    # 'mkdir -p' in Python
    if os.path.exists(path):
        return True
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_to_h5(obj, filename):
    print("Saving '" + filename + "'...")
    if os.path.exists(filename):
        os.remove(filename)
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset(filename, data=obj)


def load_from_h5(filename):
    if os.path.exists(filename):
        print("Loading '" + filename + "'...")
        with h5py.File(filename, 'r') as hf:
            return hf[filename][:]


def get_timestamps(df):
    return df.index.unique().tolist()


def load_points(points_filepath):
    df = pd.DataFrame.from_csv(points_filepath)
    return df.groupby(df.index), get_timestamps(df)


def load_tracklet(folder_path, xml_filename='tracklet_labels.xml'):
    xml_filepath = os.path.join(folder_path, xml_filename)
    return ih.read_label_from_xml(xml_filepath)


def load_camera_timestamps(folder_path, csv_filename="cap_camera.csv"):
    csv_filepath = os.path.join(folder_path, csv_filename)
    return sorted(list(pd.DataFrame.from_csv(csv_filepath).index))


def tracklet_frame_to_timestamp(frame_id, camera_timestamps):
    return camera_timestamps[frame_id]


def timestamp_to_tracklet_frame(timestamp, camera_timestamps):
    frame_id = 0

    for camera_timestamp in camera_timestamps:
        if camera_timestamp >= timestamp:
            break
        frame_id += 1

    if frame_id + 1 < len(camera_timestamps):
        if np.abs(timestamp - camera_timestamps[frame_id + 1]) < np.abs(timestamp - camera_timestamps[frame_id]):
            return frame_id + 1

    return frame_id


def get_first_tracklet_for(timestamp, tracklets, camera_timestamps):
    frame_id = timestamp_to_tracklet_frame(timestamp, camera_timestamps)
    return frame_id, tracklets[frame_id]


def save_pointclouds_with_frame_ids(input_folder_path):
    point_file = os.path.join(input_folder_path, "points.csv")
    points_out_folder = os.path.join(input_folder_path, "points")
    mkdir_p(points_out_folder)

    camera_timestamps = load_camera_timestamps(input_folder_path)
    points_df, df_timestamps = load_points(point_file)

    print("Loaded %s camera timestamps" % (len(camera_timestamps)))

    out_count = 0
    for second, group in points_df:
        timestamp = group.axes[0][0]
        frame_id = timestamp_to_tracklet_frame(timestamp, camera_timestamps)

        out_file = os.path.join(points_out_folder, str(frame_id) + ".pcd")

        z = pcl.PointCloud()
        z.from_list(np.array(group[['x', 'y', 'z']], dtype=np.float32))
        pcl.save(z, out_file)

        out_count += 1

    print("Saved out %s pointclouds" % out_count)


# complete_folders = ['suburu03', 'nissan05', 'suburu12', 'suburu07', 'nissan02', 'nissan03', 'suburu06', 'nissan07', 'suburu10', 'cmax01', 'nissan01', 'nissan06', 'suburu09', 'suburu11', 'suburu05', 'bmw01', 'suburu01', 'nissan04', 'suburu02', 'suburu04']

def process_pointcloud_timestamps_for_all_folders(folders_glob="/home/paperspace/Desktop/converted/car/training/*/*",
                                                  complete_folders=None):
    if complete_folders is None:
        complete_folders = []

    for data_input_folder in glob.glob(folders_glob):

        already_done = False
        if "long" in data_input_folder:
            already_done = True

        for done_name in complete_folders:
            if data_input_folder.endswith(done_name):
                already_done = True

        if already_done:
            print("Skipping '%s'" % data_input_folder)
        else:
            print("Processing '%s'" % data_input_folder)
            save_pointclouds_with_frame_ids(data_input_folder)

        print("     ")


def create_labels_from_tracklet(input_folder):
    bboxes, size = load_tracklet(input_folder)

    labels_out_folder = os.path.join(input_folder, "labels")
    mkdir_p(labels_out_folder)

    for i, bbox in six.iteritems(bboxes):
        out_file_name = os.path.join(labels_out_folder, str(i) + ".pickle")
        with open(out_file_name, 'wb') as out_file:
            pickle.dump([bbox['size'], bbox['place'], bbox['rot'][2]], out_file)


def create_labels_from_tracklet_for_all_folders(folders_glob="/home/paperspace/Desktop/converted/car/training/*/*"):
    for data_input_folder in glob.glob(folders_glob):
        print("Processing '%s'" % data_input_folder)
        create_labels_from_tracklet(data_input_folder)
        print("     ")
