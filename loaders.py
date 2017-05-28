import os
import pandas as pd
import parse_tracklet as pt
import numpy as np
import fnmatch


def deep_file_find(root_folder_path, file_pattern='*.*'):
    return [os.path.join(dirpath, f)
            for dirpath, dirnames, files in os.walk(root_folder_path)
            for f in fnmatch.filter(files, file_pattern)]


def load_points(folder_path, file_name='points.csv'):
    csv_filepath = os.path.join(folder_path, file_name)
    print("Loading csv from '%s'" % csv_filepath)
    df = pd.DataFrame.from_csv(csv_filepath)
    # df.index = pd.to_datetime(df.index)
    # Group the data per time period
    return df.groupby(df.index)  # .map(lambda t: t/50))#t.microsecond/500))


def load_tracklets(folder_path, file_name='tracklet_labels.xml'):
    xml_filepath = os.path.join(folder_path, file_name)
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


print( deep_file_find(os.path.dirname(os.__file__)) )
