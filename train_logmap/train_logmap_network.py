import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import numpy as np
from scipy.spatial.transform import Rotation as R
import tf_dataset
import pointnet_seg as model
import align

def safe_norm(x, epsilon=1e-8, axis=None):
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x) , axis=axis), epsilon))

def init_config():
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    return config

# set network parameters here
BATCH_SIZE = 64
RESIZE=True
N_ORIG_NEIGHBORS = 200
N_NEIGHBORS_DATASET = 120
N_NEIGHBORS = 30
TESTING_SHAPES = [21, 11, 26]
TRAINING_SHAPES = list(set(list(range(56))) - set(TESTING_SHAPES))
N_TRAINING_SHAPES = len(TRAINING_SHAPES)
N_TESTING_SHAPES = len(TESTING_SHAPES)
LOG_DIR = "log/log_famousthingi_logmap"

n_patches = 10000
path_records = "../data/training_data/famousthingi_logmap_patches_{}.tfrecords"

TRAINSET_SIZE = n_patches*N_TRAINING_SHAPES
VALSET_SIZE = n_patches*N_TESTING_SHAPES

def init_graph():
    config = init_config()
    with tf.device('/cpu:0'):
            train_iterator, training_dataset =tf_dataset.dataset([path_records.format(k) for k in TRAINING_SHAPES],
                                                                batch_size=BATCH_SIZE, n_patches = n_patches, n_neighbors=N_ORIG_NEIGHBORS)
            val_iterator, _ =tf_dataset.dataset([path_records.format(k) for k in TESTING_SHAPES]
                            ,batch_size=BATCH_SIZE,n_patches = n_patches,n_neighbors=N_ORIG_NEIGHBORS)
            handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)
            next_v = iterator.get_next() # THIS WILL BE USED AS OUR INPUT
    with tf.device('/gpu:'+str(0)):
        batch = tf.Variable(0)
        learning_rate = tf.placeholder(tf.float32, shape=[])
        is_training = tf.placeholder(tf.bool, shape=[])
        rotations =tf.placeholder(tf.float32, shape=[BATCH_SIZE, 3, 3])
        data = tf.placeholder(tf.float32, shape=[BATCH_SIZE,N_ORIG_NEIGHBORS*5])
        data = next_v[:,:N_ORIG_NEIGHBORS]
        data = data[:,:N_NEIGHBORS_DATASET]
        gt_map = data[:,:,3:]
        neighbor_points =data[:,:,:3]
        neighbor_points = neighbor_points - tf.tile(neighbor_points[:, 0:1, :], [1,N_NEIGHBORS_DATASET, 1])
        if RESIZE:
            diag =  safe_norm(tf.reduce_max(neighbor_points, axis = 1) - tf.reduce_min(neighbor_points, axis = 1), axis = -1)
            diag +=tf.random.normal([BATCH_SIZE], mean=0.0, stddev=0.01, dtype=tf.dtypes.float32)
            diag = tf.tile(diag[:,tf.newaxis,tf.newaxis], [1, neighbor_points.shape[1], neighbor_points.shape[2]])
            neighbor_points = tf.divide(neighbor_points, diag)
            gt_map = tf.divide(gt_map, diag[:,:,:2])

        dists = safe_norm(gt_map,axis = -1)
        geo_neighbors =tf.math.top_k(-dists, k=N_NEIGHBORS)[1]
        gt_map = tf.gather(gt_map, geo_neighbors, batch_dims=1)
        neighbor_points = tf.gather(neighbor_points, geo_neighbors, batch_dims=1)
        neighbor_points = neighbor_points - tf.tile(neighbor_points[:, 0:1, :], [1,N_NEIGHBORS, 1])
        neighbor_points = tf.transpose(tf.matmul(rotations, tf.transpose(neighbor_points, [0, 2, 1])),[0, 2, 1])

        with tf.variable_scope("learn_logmap"):
            map = model.logmap(neighbor_points, is_training,  batch_size=BATCH_SIZE, activation=tf.nn.relu)

        map = align.align(map, gt_map)
        dists = safe_norm(map, axis = -1)
        gt_dists = safe_norm(gt_map, axis = -1)
        loss_dist = tf.reduce_mean(tf.square(dists - gt_dists))
        loss_pos = tf.reduce_mean(tf.reduce_sum(tf.square(gt_map - map), axis = -1))
        loss = loss_dist + loss_pos
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.minimize(loss,global_step=batch)

        tf.summary.scalar("loss dist",loss_dist)
        tf.summary.scalar("loss pos",loss_pos)
        tf.summary.scalar("total_loss",loss)
        merged = tf.summary.merge_all()

        init = tf.global_variables_initializer()
    session = tf.Session(config=config)
    saver = tf.train.Saver()
    session.run(init)

    training_handle = session.run(train_iterator.string_handle())
    validation_handle =session.run(val_iterator.string_handle())
    test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), session.graph)
    train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), session.graph)

    ops = {"train": train,
            "loss": loss,
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
        feed_dict= {ops['learning_rate']: lr,
                    ops["is_training"]: True,
                    ops['handle']: ops['training_handle'],
                    ops['rotations']: rotations}
        to_run = [ops['merged'],ops['train'],ops['step'] ,ops['loss']]
        summary,_,global_step, loss_v=session.run(to_run, feed_dict=feed_dict)
        train_writer.add_summary(summary, global_step)
        l_loss.append(loss_v)
        if step%200 == 0:
            print("epoch {:4d} step {:5d}/{:5d} loss: {:3.4f}".format(epoch, step,n_steps, np.mean(l_loss)))
            l_loss = []

def eval_one_epoch(session, ops, test_writer, epoch):
    l_loss= []
    print('validation:')
    n_steps = VALSET_SIZE//BATCH_SIZE
    for step in range(n_steps):
        rotations = np.tile(np.eye(3)[np.newaxis, :,:], [BATCH_SIZE, 1, 1])
        feed_dict= {ops['learning_rate']: lr,
                    ops["is_training"]: False,
                    ops['handle']: ops['validation_handle'],
                    ops['rotations']: rotations}
        to_run = [ops['merged'],ops['step'] ,ops['loss']]
        summary,global_step, loss_v=session.run(to_run, feed_dict=feed_dict)
        test_writer.add_summary(summary, global_step)
        l_loss.append(loss_v)
        if step == n_steps-1:
            print("validation epoch {:4d} step {:5d}/{:5d} loss: {:3.4f}".format(epoch, step,n_steps, np.mean(l_loss)))
    return np.mean(l_loss)

if not os.path.exists("log"): os.mkdir("log")
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
lr = 0.00001#0.0001
best_loss = 1e10
session,  train_writer,test_writer, ops, saver = init_graph()
for epoch in range(500):
    save_path = saver.save(session, "{}/model.ckpt".format(LOG_DIR))
    train_one_epoch(session, ops, train_writer,  epoch)
    loss_val = eval_one_epoch(session, ops, test_writer,  epoch)
    if loss_val<best_loss:
        save_path = saver.save(session, "{}/best_model.ckpt".format(LOG_DIR))
        best_loss = loss_val
    save_path = saver.save(session, "{}/model.ckpt".format(LOG_DIR))
