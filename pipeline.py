from loaders import deep_load_folder, mkdir_p
from unwrapper import point_cloud_to_panorama
from image_utils import LaggingImage, adjust_gamma, draw_centroid, save_video
from centroids import get_first_centroid_for, unwrap_centroid
# from pipeline_utils import


import os
import six
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process so.')
parser.add_argument('-i', '--inputdir', type=str, nargs='?', dest='input_folder',
                    default="/Users/max/Desktop/carchallenge/car_out/",
                    help='Input root directory path')

parser.add_argument('-o', '--outdir', type=str, nargs='?', dest='output_folder',
                    default="/Users/max/Desktop/carchallenge/training_data/",
                    help='Output root directory path')

args = parser.parse_args()

print("Loading from '%s'" % args.input_folder)
print("Saving to '%s'" % args.output_folder)

mkdir_p(args.output_folder)

IMAGE_LAG_NUM = 1
PAST_IMAGE_WEIGHT = 0.4
ROTATION_INTERVAL = 15.0


def new_lagging_image():
    return LaggingImage(past_image_limit=IMAGE_LAG_NUM, past_image_weight=PAST_IMAGE_WEIGHT)


def new_past_images():
    if ROTATION_INTERVAL > 0:
        past_images = {}
        for display in ["intensity", "distance"]:
            if display not in past_images:
                past_images[display] = {}
            for i in range(int(360.0 / ROTATION_INTERVAL)):
                rotation_deg = float(i) * ROTATION_INTERVAL
                past_images[display][rotation_deg] = new_lagging_image()
        return past_images
    else:
        return {"intensity": new_lagging_image(),
                "distance": new_lagging_image()}


def process():
    for data_group_name, data_item_name, loaded_data in deep_load_folder(args.input_folder):
        output_folder_path = os.path.join(args.output_folder, data_group_name, data_item_name)
        mkdir_p(output_folder_path)

        past_images = new_past_images()
        frames = {"distance": [], "intensity": []}

        points_groups = loaded_data["points"]
        # tracklets = loaded_data["tracklets"]
        centroids = loaded_data["centroids"]
        df_timestamps = loaded_data["timestamps"]
        tracklet_centroids = None
        if centroids is not None:
            tracklet_centroids = centroids[0]

        for i in range(int(360.0 / ROTATION_INTERVAL)):
            frames = {"distance": {}, "intensity": {}}
            rotation_deg = float(i) * ROTATION_INTERVAL

            for second, group in points_groups:
                timestamp = group.axes[0][0]
                centroid = None
                if tracklet_centroids is not None:
                    centroid = get_first_centroid_for(timestamp, tracklet_centroids, df_timestamps)
                    # print           ("centroid: %s" % (centroid))
                    x, y, r = unwrap_centroid(centroid, rotation_deg=rotation_deg)
                    # print("%s points: %s" % (second, ','.join([str(x), str(y), str(r)])))

                img_dist, img_intensity = point_cloud_to_panorama(group,
                                                                  d_range=(0.0, 40.0),
                                                                  rotation_deg=rotation_deg)

                past_images["distance"][rotation_deg].add(img_dist)
                past_images["intensity"][rotation_deg].add(img_intensity)
                # past_images["depth_map"].add(img_depth)

                ts_start = group.axes[0][0]
                for display in ["distance", "intensity"]:
                    img = past_images[display][rotation_deg].get_lagged_image(scale_up=1.5)
                    # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                    if display == "intensity":
                        img = adjust_gamma(img, 1.2)

                    ## img = cv2.bitwise_not(img)

                    filename_base = "unwrapped_rot%s_%s_s%s" % (rotation_deg, display, second)
                    if centroid is not None:
                        tracklet_filename = "%s.txt" % filename_base
                        with open(os.path.join(output_folder_path, tracklet_filename), 'w') as tracklet_file:
                            tracklet_file.write("%s, %s, %s" % (x, y, r))

                    image_filename = "%s.png" % filename_base
                    image_unlabelled_output_folder = os.path.join(output_folder_path, "unlabelled")
                    mkdir_p(image_unlabelled_output_folder)
                    cv2.imwrite(os.path.join(image_unlabelled_output_folder, image_filename), img)

                    if rotation_deg not in frames[display]:
                        frames[display][rotation_deg] = []

                    if centroid is None:
                        frames[display][rotation_deg].append(img)
                    else:
                        img_centroid = draw_centroid(img, centroid, scale=1.5, rotation_deg=rotation_deg)
                        image_labelled_output_folder = os.path.join(output_folder_path, "labelled")
                        mkdir_p(image_labelled_output_folder)
                        cv2.imwrite(os.path.join(image_labelled_output_folder, image_filename), img_centroid)
                        frames[display][rotation_deg].append(img_centroid)

            save_video(frames, os.path.join(output_folder_path, filename_base))


process()
