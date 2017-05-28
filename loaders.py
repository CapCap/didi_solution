import os
import pandas as pd
import parse_tracklet as pt
import numpy as np
import fnmatch
import glob
import errno


# assumes data is arranged like `data_3/11_f/points.csv`

def deep_load_folder(root_folder_path, points_filename='points.csv'):
    root_folder_path = os.path.dirname(root_folder_path)
    print(root_folder_path)
    points_filepaths = deep_points_folder_find(root_folder_path, file_pattern=points_filename)

    for points_filepath in points_filepaths:
        # if '14_f' in points_filepath:
        data_group_name, data_item_name = points_filepath.split("/")[-2:]  # [-3::2]
        print("Loading folder: '%s'    ('%s', '%s')" % (points_filepath, data_group_name, data_item_name))
        yield data_group_name, data_item_name, handle_data_folder(points_filepath)
        # if data_group not in loaded_files:
        #    loaded_files[data_group] = {}
        # loaded_files[data_group][data_name] = handle_data_folder(points_filepath)

        # return loaded_files


def handle_data_folder(folder_path, points_filename='points.csv', tracklets_filename='tracklet_labels.xml'):
    points, timestamps = load_points(folder_path, points_filename)
    tracklets = load_tracklets(folder_path, tracklets_filename)
    centroids = get_centroids(tracklets)
    return {
        "points": points,
        "tracklets": tracklets,
        "centroids": centroids,
        "timestamps": timestamps,
    }


def deep_points_folder_find(root_folder_path, file_pattern='*.*'):
    paths = [os.path.dirname(os.path.join(dirpath, f))
             for dirpath, dirnames, files in os.walk(root_folder_path)
             for f in fnmatch.filter(files, file_pattern)]
    # paths = [path for path in paths if '/points/' in path]
    return paths


def get_timestamps(df):
    return df.index.unique().tolist()


def load_points(folder_path, csv_filename='points.csv'):
    csv_filepath = os.path.join(folder_path, csv_filename)
    print("Loading csv from '%s'" % csv_filepath)
    df = pd.DataFrame.from_csv(csv_filepath)
    # df.index = pd.to_datetime(df.index)
    # Group the data per time period
    return df.groupby(df.index), get_timestamps(df)  # .map(lambda t: t/50))#t.microsecond/500))


def load_tracklets(folder_path, xml_filename='tracklet_labels.xml'):
    xml_filepath = os.path.join(folder_path, xml_filename)
    print("Loading tracklets from '%s'" % xml_filepath)
    tracklets = pt.parse_xml(xml_filepath)
    for i in range(len(tracklets)):
        tracklet = tracklets[i]
        h, w, l = tracklet.size
        print("%s. %s" % ((i + 1), tracklet.object_type))
        print("    LxWxH:  %s x %s x %s" % (l, w, h))
        print("    Frame0: %s " % (tracklet.first_frame))
        print("    Frames: %s " % (tracklet.num_frames))
        print("    Start:  (%s, %s, %s) " % tuple(tracklet.trans[0]))
        print("    End:  (%s, %s, %s) " % tuple(tracklet.trans[-2]))
    return tracklets


def get_centroids(tracklets):
    centroids = []
    for tracklet in tracklets:
        tracklet_centroids = []
        for i in range(tracklet.num_frames):
            trans_xyz = tracklet.trans[i]
            obj_lwh = tracklet.size
            centroid = np.array(trans_xyz)  # + np.array(obj_lwh)/2.0
            tracklet_centroids.append(centroid)
        centroids.append(tracklet_centroids)
    return centroids


def mkdir_p(path):
    # 'mkdir -p' in Python
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
