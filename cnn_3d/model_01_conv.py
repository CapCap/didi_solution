#!/usr/bin/env python
import sys
import numpy as np
import tensorflow as tf
import input_helpers

import glob

import layers


# from layers import batch_norm, conv3DLayer, conv3D_to_output, fully_connected, \
#    loss_function, get_model, create_optimizer, lidar_generator


class CNNModel(object):
    def __init__(self, voxel_shape=(800, 800, 80), activation=tf.nn.relu, is_training=True,
                 resolution=0.1, scale=8, x=(-40, 40), y=(-40, 40), z=(-4, 4),
                 model_path=None):

        self.sess = tf.Session()
        self.graph = tf.get_default_graph()

        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        self.voxel_shape = voxel_shape
        self.activation = activation
        self.is_training = is_training
        self.resolution = resolution
        self.scale = scale
        self.model_path = model_path

        self.graph_built = False

    def predict(self, points, min_certainty=0.99):
        voxel = input_helpers.pc2voxel(points,
                                       resolution=self.resolution,
                                       x=self.x,
                                       y=self.y,
                                       z=self.z)

        voxel = voxel.reshape(1, voxel.shape[0], voxel.shape[1], voxel.shape[2], 1)

        # objectness = sess.run(model.objectness, feed_dict={voxel: voxel_x}[0, :, :, :, 0]
        coordinates = self.sess.run(self.coordinate, feed_dict={voxel: voxel})[0]
        y_pred = self.sess.run(self.y_pred, feed_dict={voxel: voxel})[0, :, :, :, 0]

        index = np.where(y_pred >= min_certainty)
        index = np.array(index)

        centers = np.vstack((index[0], np.vstack((index[1], index[2])))).transpose()
        centers = input_helpers.sphere2center(centers,
                                              resolution=self.resolution,
                                              scale=self.scale,
                                              min_value=np.array([self.x[0], self.y[0], self.z[0]]))

        corners = coordinates[index[0], index[1], index[2]].reshape(-1, 8, 3)
        corners = np.array([corners[t, :, :] + centers[t, :] for t in range(corners.shape[0])])

        return corners, centers, coordinates, y_pred

    def build_graph(self):
        if self.graph_built:
            return False

        self.input_voxel = tf.placeholder(tf.float32, [None, self.voxel_shape[0], self.voxel_shape[1], self.voxel_shape[2], 1])

        self.phase_train = None
        if self.is_training:
            self.phase_train = tf.placeholder(tf.bool, name='phase_train')

        self.layer1 = layers.conv3DLayer(self.input_voxel,
                                         input_dim=1,
                                         output_dim=16,
                                         size=(5, 5, 5),
                                         stride=[1, 2, 2, 2, 1],
                                         name="layer1",
                                         activation=self.activation,
                                         is_training=self.is_training)

        self.layer2 = layers.conv3DLayer(self.layer1,
                                         input_dim=16,
                                         output_dim=32,
                                         size=(5, 5, 5),
                                         stride=[1, 2, 2, 2, 1],
                                         name="layer2",
                                         activation=self.activation,
                                         is_training=self.is_training)

        self.layer3 = layers.conv3DLayer(self.layer2,
                                         input_dim=32,
                                         output_dim=64,
                                         size=(3, 3, 3),
                                         stride=[1, 2, 2, 2, 1],
                                         name="layer3",
                                         activation=self.activation,
                                         is_training=self.is_training)

        self.layer4 = layers.conv3DLayer(self.layer3,
                                         input_dim=64,
                                         output_dim=64,
                                         size=(3, 3, 3),
                                         stride=[1, 1, 1, 1, 1],
                                         name="layer4",
                                         activation=self.activation,
                                         is_training=self.is_training)

        self.objectness = layers.conv3D_to_output(self.layer4,
                                                  input_dim=64,
                                                  output_dim=2,
                                                  size=(3, 3, 3),
                                                  stride=[1, 1, 1, 1, 1],
                                                  name="objectness")

        self.coordinate = layers.conv3D_to_output(self.layer4,
                                                  input_dim=64,
                                                  output_dim=24,
                                                  size=(3, 3, 3),
                                                  stride=[1, 1, 1, 1, 1],
                                                  name="coordinate")

        self.y_pred = tf.nn.softmax(self.objectness, dim=-1)

        if self.is_training:
            initialized_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="3DCNN")
            self.sess.run(tf.variables_initializer(initialized_var))

        if self.model_path is not None:
            saver = tf.train.Saver()
            saver.restore(self.sess, self.model_path)
            # self.graph = tf.get_default_graph()

        self.graph_built = True


def clear_tf_graph():
    try:
        with tf.Session() as sess:
            tf.reset_default_graph()
    except:
        pass


def train(batch_num, points_glob, resolution=0.2, scale=4, lr=0.01, voxel_shape=(800, 800, 40), x=(0, 80), y=(-40, 40), z=(-2.5, 1.5), epochs=1,
          model_path=None, clear_graph=False, model_prefix="3dcnn_"):
    if clear_graph:
        clear_tf_graph()

    print("Training...")

    with tf.Session() as sess:
        model, voxel, phase_train = layers.get_model(sess,
                                                     CNNModel,
                                                     voxel_shape=voxel_shape,
                                                     activation=tf.nn.relu,
                                                     is_training=True)

        saver = tf.train.Saver()
        if model_path is not None:
            saver.restore(sess, model_path)
        # total_loss, obj_loss, cord_loss, g_map, g_cord = loss_func(model)
        total_loss, obj_loss, cord_loss, is_obj_loss, non_obj_loss, g_map, g_cord, y_pred = layers.loss_function(model)
        optimizer = layers.create_optimizer(total_loss, lr=lr)

        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            for (batch_x, batch_g_map, batch_g_cord) in layers.lidar_generator(batch_num, points_glob,
                                                                               resolution=resolution,
                                                                               scale=scale,
                                                                               x=x,
                                                                               y=y,
                                                                               z=z):
                sess.run(optimizer, feed_dict={voxel: batch_x,
                                               g_map: batch_g_map,
                                               g_cord: batch_g_cord,
                                               phase_train: True})

                cc = sess.run(cord_loss, feed_dict={voxel: batch_x, g_map: batch_g_map, g_cord: batch_g_cord, phase_train: True})
                iol = sess.run(is_obj_loss, feed_dict={voxel: batch_x, g_map: batch_g_map, g_cord: batch_g_cord, phase_train: True})
                nol = sess.run(non_obj_loss, feed_dict={voxel: batch_x, g_map: batch_g_map, g_cord: batch_g_cord, phase_train: True})
                print("Epoch:", '%04d' % (epoch + 1), "cord_loss_cost=", "{:.9f}".format(cc), "is_obj_loss_cost=", "{:.9f}".format(iol), "non_obj_loss_cost=", "{:.9f}".format(nol))

                # print("Epoch:", '%04d' % (epoch + 1), "total_loss=", "{:.9f}".format(ct), "obj_loss=", "{:.9f}".format(co), "coord_loss=", "{:.9f}".format(cc))

            print("Save epoch " + str(epoch + 1))
            checkpoint_name = model_prefix + ("%sx%sx%s_" % voxel_shape) + ("res%s_" % resolution) + ("sc%s" % scale) + "_" + str(epoch + 1) + ".ckpt"
            saver.save(sess, checkpoint_name)

        print("Training Complete!")


def test(model_path, points_path, resolution=0.2, scale=4, voxel_shape=(800, 800, 40), x=(0, 80), y=(-40, 40), z=(-2.5, 1.5), clear_graph=False):
    if clear_graph:
        clear_tf_graph()

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    pc = load_pc_from_pcd(points_path)

    voxel = pc2voxel(pc,
                     resolution=resolution,
                     x=x,
                     y=y,
                     z=z)

    voxel_x = voxel.reshape(1, voxel.shape[0], voxel.shape[1], voxel.shape[2], 1)

    with tf.Session() as sess:
        is_training = None
        model, voxel, phase_train = layers.get_model(sess,
                                                     CNNModel,
                                                     voxel_shape=voxel_shape,
                                                     activation=tf.nn.relu,
                                                     is_training=is_training)
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        # new_saver = tf.train.import_meta_graph(model_path)
        # new_saver.restore(sess, model_path)
        # new_saver.restore(sess, tf.train.latest_checkpoint('./'))

        objectness = sess.run(model.objectness, feed_dict={voxel: voxel_x})  # [0, :, :, :, 0]
        coordinate = sess.run(model.coordinate, feed_dict={voxel: voxel_x})  # [0]
        y_pred = sess.run(model.y, feed_dict={voxel: voxel_x})  # [0, :, :, :, 0]

        print("objectness.shape: ", objectness.shape)
        print("objectness.max && min: ", objectness.max(), objectness.min())
        print("y_pred.shape: ", y_pred.shape)
        print("y_pred.max && min: ", y_pred.max(), y_pred.min())

        index = np.where(y_pred >= 0.995)
        index = np.array(index)

        centers = np.vstack((index[0], np.vstack((index[1], index[2])))).transpose()

        print("centers: ", centers)
        print("centers.shape: ", centers.shape)

        centers = sphere2center(centers,
                                resolution=resolution,
                                scale=scale,
                                min_value=np.array([x[0], y[0], z[0]]))

        # corners = coordinate[index].reshape(-1, 8, 3) + centers[:, np.newaxis]
        # corners = (coordinate[index].T + centers)
        # print("corners_shape: ", corners.shape)
        print("voxels_shape: ", voxel.shape)

        # publish_pc2(pc, corners.reshape(-1, 3))

        return coordinate, objectness, centers, y_pred
