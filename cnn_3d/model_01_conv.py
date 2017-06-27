#!/usr/bin/env python
import sys
import numpy as np
import tensorflow as tf
from input_helpers import *
import glob

from layers import batch_norm, conv3DLayer, conv3D_to_output, deconv3D_to_output, fully_connected, \
    loss_function, get_model, create_optimizer, lidar_generator


class CNNModel(object):
    def __init__(self):
        pass

    def build_graph(self, voxel, activation=tf.nn.relu, is_training=True):
        self.layer1 = conv3DLayer(voxel,
                                  input_dim=1,
                                  output_dim=16,
                                  size=(5, 5, 5),
                                  stride=[1, 2, 2, 2, 1],
                                  name="layer1",
                                  activation=activation,
                                  is_training=is_training)

        self.layer2 = conv3DLayer(self.layer1,
                                  input_dim=16,
                                  output_dim=32,
                                  size=(5, 5, 5),
                                  stride=[1, 2, 2, 2, 1],
                                  name="layer2",
                                  activation=activation,
                                  is_training=is_training)

        self.layer3 = conv3DLayer(self.layer2,
                                  input_dim=32,
                                  output_dim=64,
                                  size=(3, 3, 3),
                                  stride=[1, 2, 2, 2, 1],
                                  name="layer3",
                                  activation=activation,
                                  is_training=is_training)

        self.layer4 = conv3DLayer(self.layer3,
                                  input_dim=64,
                                  output_dim=64,
                                  size=(3, 3, 3),
                                  stride=[1, 1, 1, 1, 1],
                                  name="layer4",
                                  activation=activation,
                                  is_training=is_training)

        self.objectness = conv3D_to_output(self.layer4,
                                           input_dim=64,
                                           output_dim=2,
                                           size=(3, 3, 3),
                                           stride=[1, 1, 1, 1, 1],
                                           name="objectness",
                                           activation=None)

        self.cordinate = conv3D_to_output(self.layer4,
                                          input_dim=64,
                                          output_dim=24,
                                          size=(3, 3, 3),
                                          stride=[1, 1, 1, 1, 1],
                                          name="cordinate",
                                          activation=None)

        self.y = tf.nn.softmax(self.objectness, dim=-1)

        # def build_graph(self, voxel, activation=tf.nn.relu, is_training=True):
        #     self.layer1 = conv3DLayer(voxel, 1, 10, 5, 5, 5, [1, 2, 2, 2, 1], name="layer1", activation=activation, is_training=is_training)
        #     self.layer2 = conv3DLayer(self.layer1, 10, 20, 5, 5, 5, [1, 2, 2, 2, 1], name="layer2", activation=activation, is_training=is_training)
        #     self.layer3 = conv3DLayer(self.layer2, 20, 30, 3, 3, 3, [1, 2, 2, 2, 1], name="layer3", activation=activation, is_training=is_training)
        #     base_shape = self.layer2.get_shape().as_list()
        #     obj_output_shape = [tf.shape(self.layer3)[0], base_shape[1], base_shape[2], base_shape[3], 2]
        #     cord_output_shape = [tf.shape(self.layer3)[0], base_shape[1], base_shape[2], base_shape[3], 24]
        #     self.objectness = deconv3D_to_output(self.layer3, 30, 2, 3, 3, 3, [1, 2, 2, 2, 1], obj_output_shape, name="objectness", activation=None)
        #     self.cordinate = deconv3D_to_output(self.layer3, 30, 24, 3, 3, 3, [1, 2, 2, 2, 1], cord_output_shape, name="cordinate", activation=None)
        #     self.y = tf.nn.softmax(self.objectness, dim=-1)


def train(batch_num, points_glob, labels_glob, resolution=0.2, scale=4, lr=0.01, voxel_shape=(800, 800, 40), x=(0, 80), y=(-40, 40), z=(-2.5, 1.5), epochs=1):
    print("Training...")
    print (points_glob, labels_glob, resolution=resolution, scale=scale, x=x, y=y, z=z)
    return
    with tf.Session() as sess:
        model, voxel, phase_train = get_model(sess,
                                              CNNModel,
                                              voxel_shape=voxel_shape,
                                              activation=tf.nn.relu,
                                              is_training=True)

        saver = tf.train.Saver()
        total_loss, obj_loss, cord_loss, is_obj_loss, non_obj_loss, g_map, g_cord, y_pred = loss_function(model)
        optimizer = create_optimizer(total_loss, lr=lr)

        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            for (batch_x, batch_g_map, batch_g_cord) in lidar_generator(batch_num, points_glob, labels_glob,
                                                                        resolution=resolution,
                                                                        scale=scale,
                                                                        x=x,
                                                                        y=y,
                                                                        z=z):
                # print( batch_x.shape, batch_g_map.shape, batch_g_cord.shape, batch_num )
                # print("batch_x.shape: ", batch_x.shape )
                # print("batch_g_map.shape: ", batch_g_map.shape )
                # print("batch_g_cord.shape: ", batch_g_cord.shape )
                sess.run(optimizer, feed_dict={voxel: batch_x,
                                               g_map: batch_g_map,
                                               g_cord: batch_g_cord,
                                               phase_train: True})
                # ct = sess.run(total_loss, feed_dict={voxel: batch_x, g_map: batch_g_map, g_cord: batch_g_cord, phase_train:True})
                # co = sess.run(obj_loss, feed_dict={voxel: batch_x, g_map: batch_g_map, g_cord: batch_g_cord, phase_train:True})
                cc = sess.run(cord_loss, feed_dict={voxel: batch_x, g_map: batch_g_map, g_cord: batch_g_cord, phase_train: True})
                iol = sess.run(is_obj_loss, feed_dict={voxel: batch_x, g_map: batch_g_map, g_cord: batch_g_cord, phase_train: True})
                nol = sess.run(non_obj_loss, feed_dict={voxel: batch_x, g_map: batch_g_map, g_cord: batch_g_cord, phase_train: True})
                # print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(ct))
                # print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(co))
                print("Epoch:", '%04d' % (epoch + 1), "cord_loss_cost=", "{:.9f}".format(cc), "is_obj_loss_cost=", "{:.9f}".format(iol), "non_obj_loss_cost=", "{:.9f}".format(nol))

            if (epoch > 0) and (epoch % 10 == 0):
                print("Save epoch " + str(epoch))
                saver.save(sess, "3dcnn_" + str(epoch) + ".ckpt")

        print("Training Complete!")


def test(points_path, resolution=0.2, scale=4, voxel_shape=(800, 800, 40), x=(0, 80), y=(-40, 40), z=(-2.5, 1.5)):
    pc = load_pc_from_pcd(points_path)

    voxel = pc2voxel(pc,
                     resolution=resolution,
                     x=x,
                     y=y,
                     z=z)

    voxel_x = voxel.reshape(1, voxel.shape[0], voxel.shape[1], voxel.shape[2], 1)

    with tf.Session() as sess:
        is_training = None
        model, voxel, phase_train = get_model(sess,
                                              CNNModel,
                                              voxel_shape=voxel_shape,
                                              activation=tf.nn.relu,
                                              is_training=is_training)
        saver = tf.train.Saver()
        # new_saver = tf.train.import_meta_graph("3dcnn_1.ckpt.meta")
        last_model = "./3dcnn_1.ckpt"
        saver.restore(sess, last_model)

        objectness = model.objectness
        cordinate = model.cordinate
        y_pred = model.y
        objectness = sess.run(objectness, feed_dict={voxel: voxel_x})[0, :, :, :, 0]
        cordinate = sess.run(cordinate, feed_dict={voxel: voxel_x})[0]
        y_pred = sess.run(y_pred, feed_dict={voxel: voxel_x})[0, :, :, :, 0]
        print(objectness.shape, objectness.max(), objectness.min())
        print(y_pred.shape, y_pred.max(), y_pred.min())

        index = np.where(y_pred >= 0.995)
        z = np.vstack((index[0], np.vstack((index[1], index[2])))).transpose()
        print(z)
        print(z.shape)

        centers = np.vstack((index[0], np.vstack((index[1], index[2])))).transpose()
        centers = sphere2center(centers,
                                resolution=resolution,
                                scale=scale,
                                min_value=np.array([x[0], y[0], z[0]]))
        corners = cordinate[index].reshape(-1, 8, 3) + centers[:, np.newaxis]
        print("corners_shape: ", corners.shape)
        print("voxels_shape: ", voxel.shape)
        # pred_corners = corners + pred_center
        # print( pred_corners )
        publish_pc2(pc, corners.reshape(-1, 3))


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
