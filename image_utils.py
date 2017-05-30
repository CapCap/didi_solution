import numpy as np
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip, clips_array
import math

# distinct colors for drawing multiple tracklets
KELLY_COLORS_DICT = dict(
    vivid_yellow=(255, 179, 0),
    strong_purple=(128, 62, 117),
    vivid_orange=(255, 104, 0),
    very_light_blue=(166, 189, 215),
    vivid_red=(193, 0, 32),
    grayish_yellow=(206, 162, 98),
    medium_gray=(129, 112, 102),
    vivid_green=(0, 125, 52),
    strong_purplish_pink=(246, 118, 142),
    strong_blue=(0, 83, 138),
    strong_yellowish_pink=(255, 122, 92),
    strong_violet=(83, 55, 122),
    vivid_orange_yellow=(255, 142, 0),
    strong_purplish_red=(179, 40, 81),
    vivid_greenish_yellow=(244, 200, 0),
    strong_reddish_brown=(127, 24, 13),
    vivid_yellowish_green=(147, 170, 0),
    deep_yellowish_brown=(89, 51, 21),
    vivid_reddish_orange=(241, 58, 19),
    dark_olive_green=(35, 44, 22))
KELLY_COLORS_LIST = KELLY_COLORS_DICT.values()


def scale_to(array, scale_to_num, dtype=np.uint8):
    return (array * float(scale_to_num) / array.max()).astype(dtype)
    # return (((array - min_val) / np.float(max_val - min_val)) * scale_to_num).astype(dtype)


def scale_to_255(array, dtype=np.uint8):
    return scale_to(array, 255, dtype=dtype)


def show(img, file_name=None):
    fig = plt.figure(dpi=1000, frameon=False, linewidth=0)
    ax = fig.gca()
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax = plt.Axes(fig, [0., 0., 1., 1.], frameon=False)
    sp = fig.add_subplot(1, 1, 1)
    sp.imshow(img, cmap='jet')
    if file_name is not None:
        fig.savefig(file_name, bbox_inches='tight')


def array_to_image(img_points, xy_points, dimensions):
    img = np.zeros(dimensions, dtype=np.uint8)
    if len(dimensions) == 3:
        img[xy_points[1], xy_points[0]] = np.asarray([img_points, img_points, img_points]).T
    else:
        img[xy_points[1], xy_points[0]] = img_points
    return img


def scale_image(img, scale_up=2):
    height, width = img.shape[:2]
    return cv2.resize(img, (int(scale_up * width), int(scale_up * height)), interpolation=cv2.INTER_CUBIC)


def draw_centroid(img, centroid, scale=1.0, rotation_deg=0.0):
    from centroids import unwrap_centroid
    img = np.copy(img)
    x, y, r = unwrap_centroid(centroid, rotation_deg=rotation_deg)
    cv2.circle(img, (int(x * scale), int(y * scale * 1.2)), int(r), (0, 255, 0), 1)
    return img


def save_video(frames, name, fps=13):
    for rotation_deg in frames['intensity'].keys():
        intensity_animation = ImageSequenceClip(frames['intensity'][rotation_deg], fps=fps)
        distance_animation = ImageSequenceClip(frames['distance'][rotation_deg], fps=fps)
        h, w, d = frames['distance'][rotation_deg][0].shape
        blank = np.zeros([h / 2, w, d], dtype=np.uint8)
        blank[:, :] = [255, 255, 255]
        blank = ImageSequenceClip([blank] * (len(frames['intensity'][rotation_deg])), fps=fps)

        final_clip = clips_array([[intensity_animation], [blank], [distance_animation]])
        filename = "%s_rot%s.webm" % (name, rotation_deg)
        final_clip.write_videofile(filename, codec='libvpx')


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


class LaggingImage(object):
    def __init__(self, past_image_limit=2, past_image_weight=0.5):
        self.past_images = []
        self.averaged_past_image = None
        self.past_image_limit = past_image_limit
        self.past_image_weight = past_image_weight

    def add(self, img):
        self.past_images.append(img)
        if len(self.past_images) > self.past_image_limit:
            self.past_images.pop(0)

    def get_lagged_image(self, scale_up=2):
        img = self.past_images[-1]
        if len(self.past_images) > 1:
            img = cv2.addWeighted(img, (1.0 - self.past_image_weight), self.averaged_past_image, self.past_image_weight, 0.0)

        self.averaged_past_image = img

        # num_images = len(self.past_images[0:-1])
        # if num_images > 0:
        #    weights = [1.0-self.past_image_weight] + ([self.past_image_weight/num_images]*num_images)
        #    img = np.mean(([img]+self.past_images[0:-1]), axis=0, weights=weights)

        # for im in self.past_images[0:-1]:
        #    img = np.add(np.multiply(img, 1.0-self.past_image_weight), np.multiply(im, self.past_image_weight))
        img = scale_to_255(img)
        return scale_image(img, scale_up)
