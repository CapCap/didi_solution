#!/usr/bin/env python
import sys
import numpy as np
import tensorflow as tf
from input_helpers import *
import glob


def batch_norm(inputs, phase_train, decay=0.9, eps=1e-5):
    """Batch Normalization

       Args:
           inputs: input data(Batch size) from last layer
           phase_train: when you test, please set phase_train "None"
       Returns:
           output for next layer
    """
    gamma = tf.get_variable("gamma",
                            shape=inputs.get_shape()[-1],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(1.0))

    beta = tf.get_variable("beta",
                           shape=inputs.get_shape()[-1],
                           dtype=tf.float32,
                           initializer=tf.constant_initializer(0.0))

    pop_mean = tf.get_variable("pop_mean",
                               trainable=False,
                               shape=inputs.get_shape()[-1],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))

    pop_var = tf.get_variable("pop_var",
                              trainable=False,
                              shape=inputs.get_shape()[-1],
                              dtype=tf.float32,
                              initializer=tf.constant_initializer(1.0))

    axes = list(range(len(inputs.get_shape()) - 1))

    if phase_train is None:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, eps)

    else:
        batch_mean, batch_var = tf.nn.moments(inputs, axes)
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, eps)


def conv3DLayer(input_layer, input_dim, output_dim, size, stride, activation=tf.nn.relu, padding="SAME", name="", is_training=True):
    height, width, length = size
    with tf.variable_scope("conv3D" + name):  # , reuse=True):
        kernel = tf.get_variable("weights",
                                 shape=[length, height, width, input_dim, output_dim],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.01))

        b = tf.get_variable("bias",
                            shape=[output_dim],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv3d(input_layer, kernel, stride, padding=padding)

        bias = tf.nn.bias_add(conv, b)
        if activation:
            bias = activation(bias, name="activation")
        bias = batch_norm(bias, is_training)
    return bias


def conv3D_to_output(input_layer, input_dim, output_dim, size, stride, padding="SAME", name=""):
    height, width, length = size
    with tf.variable_scope("conv3D" + name):  # , reuse=True):
        kernel = tf.get_variable("weights",
                                 shape=[length, height, width, input_dim, output_dim],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.01))

        return tf.nn.conv3d(input_layer, kernel, stride, padding=padding)


def fully_connected(input_layer, shape, name="", is_training=True):
    with tf.variable_scope("fully" + name):  # , reuse=True):
        kernel = tf.get_variable("weights",
                                 shape=shape,
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.01))
        fully = tf.matmul(input_layer, kernel)
        fully = tf.nn.relu(fully)
        return batch_norm(fully, is_training)

cnn_model = None
model_voxel = None
phase_train = None
def get_model(sess, model_klass, voxel_shape=(300, 300, 300), activation=tf.nn.relu, is_training=True):
    global cnn_model
    global model_voxel
    global phase_train

    if cnn_model is not None:
        return cnn_model, model_voxel, phase_train

    model_voxel = tf.placeholder(tf.float32, [None, voxel_shape[0], voxel_shape[1], voxel_shape[2], 1])

    phase_train = None
    if is_training:
        phase_train = tf.placeholder(tf.bool, name='phase_train')

    with tf.variable_scope("3DCNN") as scope:
        cnn_model = model_klass()
        cnn_model.build_graph(model_voxel, activation=activation, is_training=phase_train)

    if is_training:
        initialized_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="3DCNN")
        sess.run(tf.variables_initializer(initialized_var))

    return cnn_model, model_voxel, phase_train

def loss_function2(model):
    g_map = tf.placeholder(tf.float32, model.coordinate.get_shape().as_list()[:4])
    obj_loss = tf.reduce_sum(tf.square(tf.subtract(model.objectness[:, :, :, :, 0], g_map)))

    g_cord = tf.placeholder(tf.float32, model.coordinate.get_shape().as_list())
    cord_diff = tf.multiply(g_map, tf.reduce_sum(tf.square(tf.subtract(model.coordinate, g_cord)), 4))
    cord_loss = tf.multiply(tf.reduce_sum(cord_diff), 0.02)

    return tf.add(obj_loss, cord_loss), obj_loss, cord_loss, g_map, g_cord

def loss_func(model):
    g_map = tf.placeholder(tf.float32, model.coordinate.get_shape().as_list()[:4])
    g_cord = tf.placeholder(tf.float32, model.coordinate.get_shape().as_list())
    object_loss = tf.multiply(g_map, model.objectness[:, :, :, :, 0])
    non_gmap = tf.subtract(tf.ones_like(g_map, dtype=tf.float32), g_map)
    nonobject_loss = tf.multiply(non_gmap, model.objectness[:, :, :, :, 1])
    # sum_object_loss = tf.add(tf.exp(object_loss), tf.exp(nonobject_loss))
    sum_object_loss = tf.exp(-tf.add(object_loss, nonobject_loss))
    # sum_object_loss = tf.exp(-nonobject_loss)
    bunbo = tf.add(tf.exp(-model.objectness[:, :, :, :, 0]), tf.exp(-model.objectness[:, :, :, :, 1]))
    obj_loss = tf.multiply(tf.reduce_sum(-tf.log(tf.div(sum_object_loss, bunbo))), 0.005)

    cord_diff = tf.multiply(g_map, tf.reduce_sum(tf.square(tf.subtract(model.coordinate, g_cord)), 4))
    cord_loss = tf.reduce_sum(cord_diff)
    return tf.add(obj_loss, cord_loss), obj_loss, cord_loss, g_map, g_cord

def loss_function(model, ero=0.00001):
    g_map = tf.placeholder(tf.float32, model.coordinate.get_shape().as_list()[:4])
    non_gmap = tf.subtract(tf.ones_like(g_map, dtype=tf.float32), g_map)

    y = model.y
    is_obj_loss = -tf.reduce_sum(tf.multiply(g_map, tf.log(y[:, :, :, :, 0] + ero)))
    non_obj_loss = tf.multiply(-tf.reduce_sum(tf.multiply(non_gmap, tf.log(y[:, :, :, :, 1] + ero))), 0.0008)
    cross_entropy = tf.add(is_obj_loss, non_obj_loss)
    obj_loss = cross_entropy
    
    g_cord = tf.placeholder(tf.float32, model.coordinate.get_shape().as_list())
    cord_diff = tf.multiply(g_map, tf.reduce_sum(tf.square(tf.subtract(model.coordinate, g_cord)), 4))
    cord_loss = tf.multiply(tf.reduce_sum(cord_diff), 0.02)
    return tf.add(obj_loss, cord_loss), obj_loss, cord_loss, is_obj_loss, non_obj_loss, g_map, g_cord, y


def create_optimizer(all_loss, lr=0.001):
    opt = tf.train.AdamOptimizer(lr)
    return opt.minimize(all_loss)


def lidar_generator(batch_num, points_glob, resolution=0.2, scale=4, x=(0, 80), y=(-40, 40), z=(-2.5, 1.5)):
    import sklearn
    points_paths = glob.glob(points_glob)
    points_paths = sklearn.utils.shuffle(points_paths)

    iter_num = len(points_paths) // batch_num

    for itn in range(iter_num):
        batch_voxel = []
        batch_g_map = []
        batch_g_cord = []
        batch_start = itn * batch_num
        batch_end = (itn + 1) * batch_num

        for points_path in points_paths[batch_start:batch_end]:

            label_path = get_label_path_for_point_path(points_path)
            if not os.path.exists(label_path):
                continue
            # print("point path: %s"%points_path)

            if not os.path.exists(label_path):
                continue

            # print("label path: %s"%label_path)
            places, rots, size = read_labels(label_path)
            if places is None or len(places.shape) == 0:
                print("places is none for: %s, %s" % (label_path, points_path))
                continue

            if  ( places[0][0] < x[0] or places[0][0] > x[1] ) or ( places[0][1] < y[0] or places[0][1] > y[1] ) or ( places[0][2] < z[0] or places[0][2] > z[1] ):
                print("places (%s) out of bounds for: %s, %s" % (places[0], label_path, points_path))
                continue
                
            pc = load_pc_from_pcd(points_path)

            corners = get_boxcorners(places, rots, size)

            voxel = pc2voxel(pc,
                             resolution=resolution,
                             x=x,
                             y=y,
                             z=z)

            center_sphere, corner_label = create_label(places, size, corners,
                                                       resolution=resolution,
                                                       x=x,
                                                       y=y,
                                                       z=z,
                                                       scale=scale,
                                                       min_value=[x[0], y[0], z[0]])

            if not center_sphere.shape[0]:
                # print( "not center_sphere.shape[0]" )
                continue
                # else:
                # print( "yes center_sphere.shape[0]" )

            # import pdb
            # pdb.set_trace()

            g_map = create_objectness_label(center_sphere,
                                            resolution=resolution,
                                            x=(x[1] - x[0]),
                                            y=(y[1] - y[0]),
                                            z=(z[1] - z[0]),
                                            scale=scale)

            g_cord = corner_label.reshape(corner_label.shape[0], -1)
            g_cord = corner_to_voxel(voxel.shape, g_cord, center_sphere, scale=scale)

            batch_voxel.append(voxel)
            batch_g_map.append(g_map)
            batch_g_cord.append(g_cord)

        batch_voxel = np.array(batch_voxel, dtype=np.float32)
        if batch_voxel.shape[0] == 0:
            continue
        try:
            batch_voxel = batch_voxel[:, :, :, :, np.newaxis]
        except:
            continue

        yield batch_voxel, np.array(batch_g_map, dtype=np.float32), np.array(batch_g_cord, dtype=np.float32)


if __name__ == '__main__':
    pcd_path = "points/*.pcd"
    label_path = "labels/*.pickle"

    # train(5, pcd_path, label_path,
    #      resolution=0.1,
    #      scale=8,
    #      voxel_shape=(800, 800, 40),
    #      x=(0, 80),
    #      y=(-40, 40),
    #      z=(-2.5, 1.5))

    test(1, pcd_path,
         resolution=0.1,
         scale=8,
         voxel_shape=(800, 800, 40),
         x=(0, 80),
         y=(-40, 40), z=(-2.5, 1.5))

