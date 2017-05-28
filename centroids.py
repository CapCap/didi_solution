import numpy as np


def get_timestamps(df):
    return df.index.unique().tolist()


def get_first_centroid_for(df, timestamp, tracklet_centroids):
    timestamps = get_timestamps(df)
    i = timestamps.index(timestamp)
    percent = (float(i) / len(timestamps))
    num_centroids = len(tracklet_centroids)
    centroid_index = int(percent * num_centroids)
    index_max = num_centroids - 1
    if centroid_index > index_max:
        centroid_index = index_max
    # print("centroid_index: %s"%centroid_index)
    return tracklet_centroids[centroid_index]
