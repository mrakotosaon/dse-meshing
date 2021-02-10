import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import tf_util

def safe_norm(x, epsilon=1e-8, axis=None):
    return tf.sqrt(tf.maximum(tf.reduce_sum(x ** 2, axis=axis), epsilon))


def classifier(point_cloud, is_training,  batch_size=8,  activation=tf.nn.relu):
    """ ConvNet baseline, input is BxNx3 gray image """
    #batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    euc_dists = safe_norm(point_cloud - tf.tile(point_cloud[:,0:1,:], [1,point_cloud.shape[1], 1]), axis = -1)[:,:,tf.newaxis]
    point_cloud = tf.concat([point_cloud, euc_dists], axis = 2)
    input_image = tf.expand_dims(point_cloud, -1)

    # CONV
    bn = False
    net = tf_util.conv2d(input_image, 64, [1,4], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv1',  activation_fn=activation)
    net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv2',  activation_fn=activation)
    net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv3',  activation_fn=activation)
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv4',  activation_fn=activation)
    points_feat1 = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv5',  activation_fn=activation)
    # MAX
    pc_feat1 = tf_util.max_pool2d(points_feat1, [num_point,1], padding='VALID', scope='maxpool1')
    # FC
    pc_feat1 = tf.reshape(pc_feat1, [batch_size, 1024])
    pc_feat1 = tf_util.fully_connected(pc_feat1, 1024, bn=bn, is_training=is_training, scope='fc1', activation_fn=activation)
    pc_feat1 = tf_util.fully_connected(pc_feat1, 1024, bn=bn, is_training=is_training, scope='fc2', activation_fn=activation)
    print(pc_feat1)

    # CONCAT
    pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [batch_size, 1, 1, -1]), [1, num_point, 1, 1])

    points_feat1_concat = tf.concat(axis=3, values=[tf.expand_dims(point_cloud, -2), pc_feat1_expand])

    # CONV
    net = tf_util.conv2d(points_feat1_concat, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv6', activation_fn=activation)
    net = tf_util.conv2d(net, 528, [1,1], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv7', activation_fn=activation)
    net = tf_util.conv2d(net, 3, [1,1], padding='VALID', stride=[1,1],
                         activation_fn=None, scope='conv8')
    euc_dists = safe_norm(net - tf.tile(net[:,0:1,:,:], [1,point_cloud.shape[1],1, 1]), axis = -1)[:,:,:,tf.newaxis]
    pc1 = net
    net = tf.concat([net, euc_dists], axis = 3)

    points_feat2_concat = tf.concat(axis=3, values=[net, pc_feat1_expand])
    net = tf_util.conv2d(points_feat2_concat, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv9', activation_fn=activation)
    net = tf_util.conv2d(net, 528, [1,1], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv10', activation_fn=activation)
    net = tf_util.conv2d(net, 1, [1,1], padding='VALID', stride=[1,1],
                         activation_fn=None, scope='conv11')


    net = tf.squeeze(net,2)
    return net


def logmap(point_cloud, is_training,  batch_size=8,  activation=tf.nn.relu):
    """ ConvNet baseline, input is BxNx3 gray image """
    num_point = point_cloud.get_shape()[1].value
    euc_dists = safe_norm(point_cloud - tf.tile(point_cloud[:,0:1,:], [1,point_cloud.shape[1], 1]), axis = -1)[:,:,tf.newaxis]
    point_cloud = tf.concat([point_cloud, euc_dists], axis = 2)
    input_image = tf.expand_dims(point_cloud, -1)
    # CONV
    bn = False
    net = tf_util.conv2d(input_image, 64, [1,4], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv1',activation_fn=activation)
    net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv2',activation_fn=activation)
    net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv3',activation_fn=activation)
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv4',activation_fn=activation)
    points_feat1 = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv5',activation_fn=activation)
    # MAX
    pc_feat1 = tf_util.max_pool2d(points_feat1, [num_point,1], padding='VALID', scope='maxpool1')
    # FC
    pc_feat1 = tf.reshape(pc_feat1, [batch_size, 1024])
    pc_feat1 = tf_util.fully_connected(pc_feat1, 1024, bn=bn, is_training=is_training, scope='fc1',activation_fn=activation)
    pc_feat1 = tf_util.fully_connected(pc_feat1, 1024, bn=bn, is_training=is_training, scope='fc2',activation_fn=activation)

    # CONCAT
    pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [batch_size, 1, 1, -1]), [1, num_point, 1, 1])

    points_feat1_concat = tf.concat(axis=3, values=[tf.expand_dims(point_cloud, -2), pc_feat1_expand])

    # CONV
    net = tf_util.conv2d(points_feat1_concat, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv6',activation_fn=activation)
    net = tf_util.conv2d(net, 528, [1,1], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv7',activation_fn=activation)
    net = tf_util.conv2d(net, 3, [1,1], padding='VALID', stride=[1,1],
                         activation_fn=None, scope='conv8')
    euc_dists = safe_norm(net - tf.tile(net[:,0:1,:,:], [1,point_cloud.shape[1],1, 1]), axis = -1)[:,:,:,tf.newaxis]
    pc1 = net
    net = tf.concat([net, euc_dists], axis = 3)

    points_feat2_concat = tf.concat(axis=3, values=[net, pc_feat1_expand])
    net = tf_util.conv2d(points_feat2_concat, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv9',activation_fn=activation)
    net = tf_util.conv2d(net, 528, [1,1], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv10',activation_fn=activation)
    net = tf_util.conv2d(net, 2, [1,1], padding='VALID', stride=[1,1],
                         activation_fn=None, scope='conv11')
    net = tf.squeeze(net,2)
    return net
