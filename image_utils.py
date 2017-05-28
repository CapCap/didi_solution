import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


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


def draw_centroid(img, centroid, scale=1.0):
    from .unwrapper import distance, unwrap_point
    img = np.copy(img)
    x, y, z = centroid[:3]
    d = distance(x, y, z)
    x, y = unwrap_point(x, y, z, d)
    # This is totally arbitrary
    r = (200 / d)
    cv2.circle(img, (int(x * scale), int(y * scale * 1.2)), int(r), (0, 255, 0), 1)
    return img


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

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
