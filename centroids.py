import numpy as np


def unwrap_centroid(centroid, rotation_deg=0.0):
    from unwrapper import distance, unwrap_point
    x, y, z = centroid[:3]
    d = distance(x, y, z)
    x, y = unwrap_point(x, y, z, d, rotation_deg=rotation_deg)
    # This is totally arbitrary
    r = (200 / d)
    return x, y, r

def get_first_centroid_for(timestamp, tracklet_centroids, df_timestamps):
    i = df_timestamps.index(timestamp)
    percent = (float(i) / len(df_timestamps))
    num_centroids = len(tracklet_centroids)
    centroid_index = int(percent * num_centroids)
    index_max = num_centroids - 1
    if centroid_index > index_max:
        centroid_index = index_max
    # print("centroid_index: %s"%centroid_index)
    return tracklet_centroids[centroid_index]
