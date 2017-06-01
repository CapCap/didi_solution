import os
import json
import cv2
import six
import numpy as np
import math
from moviepy.editor import ImageSequenceClip, clips_array, TextClip, CompositeVideoClip

from tracklets.python.generate_tracklet import Tracklet, TrackletCollection

from loaders import handle_data_folder
# from centroids import get_first_centroid_for
from unwrapper import wrap_point, unwrap_point, calculate_distance, IMG_DIMENSIONS
from image_utils import array_to_image


def load_center_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def draw_centers(img, centers, d=10):
    img = np.copy(img)
    r = 200 / d
    if centers is not None:
        for center in centers:
            cv2.circle(img, (int(math.floor(center[0])), int(math.floor(center[1]))), int(r), (255, 0, 0), 2)
            break
    return img


def load_image(image_filepath):
    with open(image_filepath, 'r') as f:
        return np.load(f)


def save_video(frames_dict, fps=13):
    frame_keys = sorted([key for key in six.iterkeys(frames_dict)])

    grid_dim = 2  # int(math.ceil(math.sqrt(len(frame_keys)) / 2.0))
    clips = []
    for frame_key in frame_keys:
        h, w, _ = frames_dict[frame_key][0].shape
        h += 40
        w += 40
        # title = TextClip(str(frame_key), color='black', fontsize=25, method='label')
        # title.set_position(("center", "top"))
        animation = ImageSequenceClip(frames_dict[frame_key], fps=fps)
        animation = animation.on_color(size=(w, h), color=(255, 255, 255))
        # animation.set_position(("center", "bottom"))
        # video = CompositeVideoClip([animation, title], size=(h, w))
        # clips.append(clips_array([[title], [animation]]))  # clips_array([[title], [animation]]))
        clips.append(animation)  # clips_array([[title], [animation]]))

    arranged_clips = []
    start = 0
    i = 0
    while start < len(clips):
        arranged_clips.append(clips[i * grid_dim:(i + 1) * grid_dim])
        i += 1
        start = i * grid_dim

    under = len(arranged_clips[0]) - len(arranged_clips[-1])
    if under > 0:
        for i in range(under):
            arranged_clips[-1].append(arranged_clips[-1][-1])
    final_clip = clips_array(arranged_clips)
    #final_clip.write_videofile("data_3_14_results_output.mp4", codec='mpeg4')
    final_clip.write_videofile("results_output.mp4", codec='libx264')


def parse_image_name(image_name):
    _, rot_str, display_type, timestamp_str = image_name.split('_')
    timestamp_str = timestamp_str.replace("s", "").replace(".png", "")
    rot = float(rot_str.replace("rot", ""))
    return rot, display_type, timestamp_str


center_json = load_center_json("center_without_false_positives_json.js")
#center_json = load_center_json("data_3_14_centers_json.js")
base_folder = "/Users/max/Desktop/carchallenge/training_data/round_1_test/19_f2/unlabelled"
#base_folder = "/Users/max/Desktop/carchallenge/training_data/data_3/14/labelled"

points_filepath = "/Users/max/Desktop/carchallenge/car_out/round_1_test/19_f2"
#points_filepath = "/Users/max/Desktop/carchallenge/car_out/data_3/14"
#data = handle_data_folder(points_filepath)

#df_points = data["points"]
image_names = sorted([key for key in six.iterkeys(center_json)])

image_frames = {}
xyz_point_dict = {}
tracklets = []

for image_name in image_names:
    rot, display_type, timestamp_str = parse_image_name(image_name)
    #if rot != 0.0:
    #    continue

    if rot not in xyz_point_dict:
        xyz_point_dict[rot] = []

    img_xy_points = center_json[image_name]
    #if len(img_xy_points) > 0:
    #    if img_xy_points[0] is not None and not isinstance(img_xy_points[0], list):
    #        img_xy_points = [img_xy_points]

    #img_xy_points = img_xy_points[0]
    #points = df_points.get_group(int(timestamp_str))
    #points = calculate_distance(points, d_range=(0.0, 40.0))

    #x_points, y_points = unwrap_point(points['x'], points['y'], points['z'], points['map_distance'], rotation_deg=rot)
    # for img_x, img_y in img_xy_points:
    #if len(img_xy_points) > 0:
    #    img_x, img_y = img_xy_points[0]
    #    img_distance = array_to_image(points['map_distance'], [x_points, y_points], IMG_DIMENSIONS)
    #    img_x = img_x / 1.5
    #    img_y = img_y / (1.5 * 1.2)
    #    d = img_distance[int(img_y)][int(img_y)]
    #    xyz_point = wrap_point(img_x, img_y, d, unrotation_deg=rot)
    #    xyz_point_dict[rot].append(xyz_point)
    #else:
    #    xyz_point_dict[rot].append(None)


    image_filepath = os.path.join(base_folder, image_name)
    img = cv2.imread(image_filepath)
    if rot not in image_frames:
       image_frames[rot] = []
    image_frames[rot].append(draw_centers(img, img_xy_points))

print("Processing Video...")
save_video(image_frames)

#tc = TrackletCollection()
#tc.tracklets = tracklets
#tc.write_xml("output_tracklets.xml")
#
#tracklet = Tracklet(object_type="car", l=5.0, w=5.0, h=2.0)
#tracklet.poses.append({
#    "tx": xyz_point[0],
#    "ty": xyz_point[1],
#    "tz": xyz_point[2],
#    "rx": 0,
#    "ry": 0,
#    "rz": 0
#})
#tracklets.append(tracklet)
#