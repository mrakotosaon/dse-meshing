import os
import sys
import trimesh
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import numpy as np
from sklearn.neighbors import KDTree
import tensorflow as tf
import importlib
from scipy.spatial.transform import Rotation as R
import tf_dataset
import models_logmap.pointnet_seg as model
N_NEIGHBORS_DATASET = 600
N_ORIG_NEIGHBORS = 600
#N_NEIGHBORS = 200#400
#N_NEAREST_NEIGHBORS = 50
import utils_network as utils
BATCH_SIZE = 32

def safe_norm(x, epsilon=1e-8, axis=None):
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x) , axis=axis), epsilon))

def init_config():
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    return config

BN_INIT_DECAY = 0.1
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = 10300*BATCH_SIZE*5
BN_DECAY_CLIP = 0.99

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

n_patches = 4000
# N_TESTING_SHAPES = 1#5
# N_TRAINING_SHAPES = 1#270
# VALSET_SIZE = n_patches*N_TESTING_SHAPES
# TRAINSET_SIZE = n_patches*N_TRAINING_SHAPES
# LOG_DIR = "/media/disk1/mj_data/mesh_generation/logmaps_models/log_planes_double_network_new_data"
# path_records = "/media/disk1/mj_data/mesh_generation/planes_geo_dataset_new_shapes/planes_logmap_patches_{}.tfrecords"


# N_TRAINING_SHAPES = 25#270
# N_TESTING_SHAPES = 2#5
# VALSET_SIZE = n_patches*N_TESTING_SHAPES
# TRAINSET_SIZE = n_patches*N_TRAINING_SHAPES
# LOG_DIR = "/media/disk1/mj_data/mesh_generation/latest_logmaps_models/log_pcpnet_retrained"
# path_records = "/media/disk1/mj_data/mesh_generation/pcpnet_dataset/planes_logmap_patches_{}.tfrecords"
RESIZE = True



N_NEIGHBORS = 120#150#120#40#400
N_NEAREST_NEIGHBORS = 30#40#30
TRAINING_SHAPES = [0,1,3,4,5,6,7,8,9,10,11,13,14,15,16,18,19,20,21,22,23,24,26,25]
TESTING_SHAPES = [12, 2, 17]
N_TRAINING_SHAPES = len(TRAINING_SHAPES)
N_TESTING_SHAPES = len(TESTING_SHAPES)
path_records = "/media/disk1/mj_data/mesh_generation/pcpnet_dataset/planes_logmap_patches_{}.tfrecords"
LOG_DIR = "/media/disk1/mj_data/mesh_generation/latest_logmaps_models/log_pcpnet_retrained_resize_30nn_v2"
TRAINSET_SIZE = n_patches*N_TRAINING_SHAPES
VALSET_SIZE = n_patches*N_TESTING_SHAPES


def init_graph():
    config = init_config()
    with tf.device('/cpu:0'):
            # train_iterator, training_dataset =tf_dataset.dataset([path_records.format(k) for k in range(N_TRAINING_SHAPES)],
            #                                                     batch_size=BATCH_SIZE, n_patches =n_patches,n_neighbors=N_ORIG_NEIGHBORS)
            # #val_iterator, _ =tf_dataset.dataset([path_records.format(k) for k in range(N_TESTING_SHAPES)]
            # val_iterator, _ =tf_dataset.dataset([path_records.format(k) for k in range(N_TRAINING_SHAPES,N_TRAINING_SHAPES+N_TESTING_SHAPES)]
            #                     ,batch_size=BATCH_SIZE, n_patches=n_patches,n_neighbors=N_ORIG_NEIGHBORS)

            train_iterator, training_dataset =tf_dataset.dataset([path_records.format(k) for k in TRAINING_SHAPES],
                                                                batch_size=BATCH_SIZE, n_patches = n_patches, n_neighbors=N_ORIG_NEIGHBORS)
            val_iterator, _ =tf_dataset.dataset([path_records.format(k) for k in TESTING_SHAPES]
                            ,batch_size=BATCH_SIZE,n_patches = n_patches,n_neighbors=N_ORIG_NEIGHBORS)

            handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)
            next = iterator.get_next() # THIS WILL BE USED AS OUR INPUT
    with tf.device('/gpu:'+str(0)):
        batch = tf.Variable(0)
        learning_rate = tf.placeholder(tf.float32, shape=[])
        is_training = tf.placeholder(tf.bool, shape=[])
        is_training_geo_dist = tf.constant(False,dtype = tf.bool)

        bn_decay = get_bn_decay(batch)
        rotations =tf.placeholder(tf.float32, shape=[BATCH_SIZE, 3, 3])
        data = tf.placeholder(tf.float32, shape=[BATCH_SIZE,N_NEIGHBORS_DATASET*5])
        data = next
        neighbor_points =data[:,:N_NEIGHBORS,:3]
        gt_map = data[:,:N_NEIGHBORS,3:]
        neighbor_points = neighbor_points - tf.tile(neighbor_points[:, 0:1, :], [1,N_NEIGHBORS, 1])
        if RESIZE:
            # resize neighborhood
            diag =  safe_norm(tf.reduce_max(neighbor_points, axis = 1) - tf.reduce_min(neighbor_points, axis = 1), axis = -1)
            diag +=tf.random.normal([BATCH_SIZE], mean=0.0, stddev=0.01, dtype=tf.dtypes.float32)
            diag = tf.tile(diag[:,tf.newaxis,tf.newaxis], [1, neighbor_points.shape[1], neighbor_points.shape[2]])
            neighbor_points = tf.divide(neighbor_points, diag)
            gt_map = tf.divide(gt_map, diag[:,:,:2])

        neighbor_points = tf.transpose(tf.matmul(rotations, tf.transpose(neighbor_points, [0, 2, 1])),[0, 2, 1])
        with tf.variable_scope("learn_geo_dist"):
            #geo_distances = model.network_fold_distances_v2(neighbor_points, is_training_geo_dist,   batch_size=BATCH_SIZE)
            geo_distances = model.network_fold_distances_v3(neighbor_points, is_training_geo_dist,   batch_size=BATCH_SIZE)
        geo_distances = tf.squeeze(geo_distances)
        closests = tf.math.top_k(tf.math.sigmoid(geo_distances), k=N_NEAREST_NEIGHBORS)[1]
        neighbor_points = tf.gather(neighbor_points, closests, batch_dims=1)
        gt_map = tf.gather(gt_map, closests, batch_dims=1)

        with tf.variable_scope("learn_logmap"):
            map = model.network_fold_double_high_capacity_v2(neighbor_points, is_training, bn_decay=None, batch_size=BATCH_SIZE)

        map = utils.align(map, gt_map)
        dists = safe_norm(map, axis = -1)
        gt_dists = safe_norm(gt_map, axis = -1)
        loss_dist = tf.reduce_mean(tf.square(dists - gt_dists))
        loss_pos = tf.reduce_mean(tf.reduce_sum(tf.square(gt_map - map), axis = -1))
        loss = loss_dist + loss_pos
        optimizer = tf.train.AdamOptimizer(learning_rate)
        learn_logmap_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='learn_logmap')
        train = optimizer.minimize(loss,global_step=batch, var_list = learn_logmap_vars)
        tf.summary.scalar("loss dist",loss_dist)
        tf.summary.scalar("loss pos",loss_pos)
        tf.summary.scalar('bn_decay', bn_decay)
        tf.summary.scalar("total_loss",loss)
        merged = tf.summary.merge_all()
        learn_logmap_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='learn_logmap')
        saver_pretrain= tf.train.Saver(var_list = learn_logmap_var)
        learn_geo_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='learn_geo_dist')
        geo_pretrain= tf.train.Saver(var_list = learn_geo_var)

        init = tf.global_variables_initializer()
    session = tf.Session(config=config)
    session.run(init)
    #geo_pretrain.restore(session,"/media/disk1/mj_data/mesh_generation/logmaps_models/log_planes_geo_distances/model.ckpt")
    #geo_pretrain.restore(session,"/media/disk1/mj_data/mesh_generation/logmaps_models/log_planes_geo_distances_100/model.ckpt")
    # saver_pretrain.restore(session,"/media/disk1/mj_data/mesh_generation/logmaps_models/log_planes_logmap_N100_from_geo_100_2/model.ckpt")
    # geo_pretrain.restore(session,"/media/disk1/mj_data/mesh_generation/logmaps_models/log_planes_logmap_N100_from_geo_100_2/model.ckpt")
    #saver_pretrain.restore(session,"/media/disk1/mj_data/mesh_generation/logmaps_models/log_planes_logmap_N100_new_dataset/model.ckpt")
    #geo_pretrain.restore(session,"/media/disk1/mj_data/mesh_generation/logmaps_models/log_planes_geo_distances_new_shapes_classification_high_lr2/model.ckpt")
    #saver_pretrain.restore(session,"/media/disk1/mj_data/mesh_generation/latest_logmaps_models/log_pcpnet_logmap_resize/best_model.ckpt")
    #geo_pretrain.restore(session,"/media/disk1/mj_data/mesh_generation/latest_logmaps_models/log_pcpnet_geo_distances_resize/best_model.ckpt")

    saver_pretrain.restore(session,"/media/disk1/mj_data/mesh_generation/latest_logmaps_models/log_pcpnet_logmap_resize_30nn/best_model.ckpt")
    #geo_pretrain.restore(session,"/media/disk1/mj_data/mesh_generation/latest_logmaps_models/log_pcpnet_geo_distances_resize_40nn/best_model.ckpt")
    geo_pretrain.restore(session,"/media/disk1/mj_data/mesh_generation/latest_logmaps_models/log_pcpnet_geo_distances_resize_30nn_v2/best_model.ckpt")

    training_handle = session.run(train_iterator.string_handle())
    validation_handle =session.run(val_iterator.string_handle())
    test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test2'), session.graph)
    train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train2'), session.graph)
    saver = tf.train.Saver()
    #saver.restore(session,"/media/disk1/mj_data/mesh_generation/logmaps_models/log_planes_logmap_N100_from_geo_100/model.ckpt")

    ops = {"train": train,
            "loss": loss,
            "loss_dist": loss_dist,
            "loss_dist": loss_dist,
            'res_map': map,
            "rotations": rotations,
            "step": batch,
            'is_training': is_training,
            "merged": merged,
            'handle':handle,
            'learning_rate' : learning_rate,
            'training_handle': training_handle,
            'validation_handle': validation_handle}
    return session,   train_writer,test_writer, ops, saver


def train_one_epoch(session, ops, train_writer, epoch):
    l_loss= []
    n_steps =  TRAINSET_SIZE//BATCH_SIZE
    for step in range(n_steps):
        rotations = R.random(BATCH_SIZE).as_matrix()
        feed_dict= {ops['learning_rate']: lr, ops["is_training"]: True,ops['handle']: ops['training_handle'],
                    ops['rotations']: rotations}
        to_run = [ops['merged'],ops['train'],ops['step'] ,ops['loss'], ops['res_map']]
        summary,_,global_step, loss_v, map_v=session.run(to_run, feed_dict=feed_dict)
        train_writer.add_summary(summary, global_step)
        l_loss.append(loss_v)
        if step%100 == 99:
            print("epoch {} step {}/{}  loss :  {}".format(epoch, step,n_steps, np.mean(l_loss)))
            l_loss = []

def eval_one_epoch(session, ops, test_writer, epoch):
    l_loss= []
    n_steps = VALSET_SIZE//BATCH_SIZE
    for step in range(n_steps):
        rotations = np.tile(np.eye(3)[np.newaxis, :,:], [BATCH_SIZE, 1, 1])
        feed_dict= {ops['learning_rate']: lr, ops["is_training"]: False,ops['handle']: ops['validation_handle'],
                    ops['rotations']: rotations}
        to_run = [ops['merged'],ops['step'] ,ops['loss'], ops['res_map']]
        summary,global_step, loss_v, map_v=session.run(to_run, feed_dict=feed_dict)
        test_writer.add_summary(summary, global_step)
        l_loss.append(loss_v)
        if step%10 == 9:
            print("validation epoch {} step {}/{}  loss :  {}".format(epoch, step,n_steps, np.mean(l_loss)))
            #l_loss = []
    return np.mean(l_loss)

#LOG_DIR = "/media/disk1/mj_data/mesh_generation/logmaps_models/log_planes_patches_logmap_coherent_data_augm_400_point_features_test2_large_batch_size32_3layers_latest"
##LOG_DIR = "/media/disk1/mj_data/mesh_generation/logmaps_models/log_planes_patches_logmap_coherent_data_augm_400_point_features_test2_large_batch_size64_3layers"
#LOG_DIR = "/media/disk1/mj_data/mesh_generation/logmaps_models/log_planes_logmap_N100_from_geo_100_2"

if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
lr = 0.00001#0.000001#0.0001#0.0000001#0.00001#0.0001 #0.00001
best_loss = 1e10
session,  train_writer,test_writer, ops, saver = init_graph()
for epoch in range(600):
    train_one_epoch(session, ops, train_writer,  epoch)
    loss_val = eval_one_epoch(session, ops, test_writer,  epoch)
    if loss_val<best_loss:
        save_path = saver.save(session, "{}/best_model.ckpt".format(LOG_DIR))
        best_loss = loss_val
    save_path = saver.save(session, "{}/model.ckpt".format(LOG_DIR))
