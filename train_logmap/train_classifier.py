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
N_NEAREST_NEIGHBORS = 30
N_NEIGHBORS = 120
TESTING_SHAPES = [21, 11, 26]
TRAINING_SHAPES = list(set(list(range(56))) - set(TESTING_SHAPES))
N_TRAINING_SHAPES = len(TRAINING_SHAPES)
print(N_TRAINING_SHAPES)
N_TESTING_SHAPES = len(TESTING_SHAPES)
LOG_DIR = "log/log_famousthingi_classifier"
n_patches = 10000
path_records = "../data/training_data/famousthingi_logmap_patches_{}.tfrecords"


TRAINSET_SIZE = n_patches*N_TRAINING_SHAPES
VALSET_SIZE = n_patches*N_TESTING_SHAPES


def init_graph():
    config = init_config()
    with tf.device('/cpu:0'):
            train_iterator, training_dataset =tf_dataset.dataset([path_records.format(k) for k in TRAINING_SHAPES],
                                                                batch_size=BATCH_SIZE, n_patches=n_patches, n_neighbors=N_ORIG_NEIGHBORS)

            val_iterator, _ =tf_dataset.dataset([path_records.format(k) for k in TESTING_SHAPES]
                            ,batch_size=BATCH_SIZE,n_patches=n_patches,n_neighbors=N_ORIG_NEIGHBORS)


            handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)
            next = iterator.get_next() # THIS WILL BE USED AS OUR INPUT
    with tf.device('/gpu:'+str(0)):
        batch = tf.Variable(0)
        learning_rate = tf.placeholder(tf.float32, shape=[])
        is_training = tf.placeholder(tf.bool, shape=[])
        rotations =tf.placeholder(tf.float32, shape=[BATCH_SIZE, 3, 3])
        data = tf.placeholder(tf.float32, shape=[BATCH_SIZE,N_ORIG_NEIGHBORS*5])
        data = next[:,:N_ORIG_NEIGHBORS]

        data = data[:,:N_NEIGHBORS]
        neighbor_points =data[:,:N_NEIGHBORS,:3]
        gt_map = data[:,:N_NEIGHBORS,3:]
        neighbor_points = neighbor_points - tf.tile(neighbor_points[:, 0:1, :], [1,N_NEIGHBORS, 1])
        neighbor_points = tf.transpose(tf.matmul(rotations, tf.transpose(neighbor_points, [0, 2, 1])),[0, 2, 1])

        if RESIZE:
            diag =  safe_norm(tf.reduce_max(neighbor_points, axis = 1) - tf.reduce_min(neighbor_points, axis = 1), axis = -1)
            diag = tf.tile(diag[:,tf.newaxis,tf.newaxis], [1, neighbor_points.shape[1], neighbor_points.shape[2]])
            neighbor_points = tf.divide(neighbor_points, diag)
            gt_map = tf.divide(gt_map, diag[:,:,:2])

        with tf.variable_scope("learn_geo_dist"):
            map = model.classifier(neighbor_points, is_training,  batch_size=BATCH_SIZE, activation=tf.nn.relu)
        map = tf.squeeze(map)
        gt_dists = safe_norm(gt_map, axis = -1)

        geo_neighbors =tf.math.top_k(-gt_dists, k=N_NEAREST_NEIGHBORS)[1]
        geo_neighbors_indices =tf.stack([tf.tile(tf.range(BATCH_SIZE)[:, tf.newaxis], [ 1, N_NEAREST_NEIGHBORS]), geo_neighbors], axis = 2)
        geo_neighbors_indices = tf.reshape(geo_neighbors_indices, [-1, 2])
        labels = tf.cast(tf.scatter_nd(geo_neighbors_indices, tf.reshape(tf.ones_like(geo_neighbors), [-1]), map.shape), tf.float32)
        class_loss = tf.reduce_mean(tf.square(labels - tf.nn.sigmoid(map)))*30.0

        predicted_geo_neighbors =tf.math.top_k(tf.nn.sigmoid(map), k=N_NEAREST_NEIGHBORS)[1]
        predicted_geo_neighbors_indices =tf.stack([tf.tile(tf.range(BATCH_SIZE)[ :,tf.newaxis], [1, N_NEAREST_NEIGHBORS]), predicted_geo_neighbors], axis = 2)
        predicted_geo_neighbors_indices = tf.reshape(predicted_geo_neighbors_indices, [-1, 2])
        predicted_labels = tf.cast(tf.scatter_nd(predicted_geo_neighbors_indices, tf.reshape(tf.ones_like(predicted_geo_neighbors), [-1]), map.shape), dtype=tf.float32)
        accuracy = 1 - tf.reduce_mean(tf.abs(labels- predicted_labels))


        loss_dist = tf.reduce_mean(tf.square(map - gt_dists))*5.0
        loss = class_loss
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.minimize(loss,global_step=batch)
        tf.summary.scalar("loss dist",loss_dist)
        tf.summary.scalar("class loss",class_loss)
        tf.summary.scalar("accuracy ",accuracy)
        tf.summary.scalar("total_loss",loss)
        merged = tf.summary.merge_all()
        learn_geo_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='learn_geo_dist')
        geo_pretrain= tf.train.Saver(var_list = learn_geo_var)
        init = tf.global_variables_initializer()
    session = tf.Session(config=config)
    session.run(init)

    training_handle = session.run(train_iterator.string_handle())
    validation_handle =session.run(val_iterator.string_handle())
    test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test2'), session.graph)
    train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train2'), session.graph)
    saver = tf.train.Saver()

    ops = {"train": train,
            "loss": loss,
            "rotations": rotations,
            'data':data,
            "step": batch,
            'is_training': is_training,
            "merged": merged,
            "accuracy":accuracy,
            'handle':handle,
            'learning_rate' : learning_rate,
            'training_handle': training_handle,
            'validation_handle': validation_handle}
    return session,   train_writer,test_writer, ops, saver


def train_one_epoch(session, ops, train_writer, epoch):
    l_loss= []
    l_acc= []
    n_steps =  TRAINSET_SIZE//BATCH_SIZE
    for step in range(n_steps):
        rotations = R.random(BATCH_SIZE).as_matrix()
        feed_dict= {ops['learning_rate']: lr,
                    ops["is_training"]: True,
                    ops['handle']: ops['training_handle'],
                    ops['rotations']: rotations}
        to_run = [ops['merged'],ops['train'],ops['step'],ops['loss'], ops['accuracy']]
        summary,_,global_step, loss_v, accuracy_v=session.run(to_run, feed_dict=feed_dict)
        train_writer.add_summary(summary, global_step)
        l_loss.append(loss_v)
        l_acc.append(accuracy_v)
        if step%200 == 0:
            print("epoch {:4d} step {:5d}/{:5d} loss: {:3.4f} accuracy: {:3.2f}%".format(epoch, step,n_steps, np.mean(l_loss), np.mean(l_acc)*100))
            l_loss = []
            l_acc = []

def eval_one_epoch(session, ops, test_writer, epoch):
    l_loss= []
    l_acc= []
    print('validation:')    
    n_steps = VALSET_SIZE//BATCH_SIZE
    for step in range(n_steps):
        rotations = np.tile(np.eye(3)[np.newaxis, :,:], [BATCH_SIZE, 1, 1])
        feed_dict= {ops['learning_rate']: lr,
                    ops["is_training"]: False,
                    ops['handle']: ops['validation_handle'],
                    ops['rotations']: rotations}
        to_run = [ops['merged'],ops['step'] ,ops['loss'], ops['accuracy']]
        summary,global_step, loss_v,accuracy_v=session.run(to_run, feed_dict=feed_dict)
        test_writer.add_summary(summary, global_step)
        l_loss.append(loss_v)
        l_acc.append(accuracy_v)
        if step==n_steps-1:
            print("validation epoch {:4d} step {:5d}/{:5d} loss: {:3.4f} accuracy: {:3.2f}%".format(epoch, step,n_steps, np.mean(l_loss), np.mean(l_acc)*100))
    return np.mean(l_loss)


if __name__ == '__main__':
    if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
    lr = 0.0005
    best_loss = 1e10
    session,  train_writer,test_writer, ops, saver = init_graph()
    for epoch in range(1000):
        if epoch>20:
            lr = 0.00001
        train_one_epoch(session, ops, train_writer,  epoch)
        loss_val = eval_one_epoch(session, ops, test_writer,  epoch)
        if loss_val<best_loss:
            save_path = saver.save(session, "{}/best_model.ckpt".format(LOG_DIR))
            best_loss = loss_val
        save_path = saver.save(session, "{}/model.ckpt".format(LOG_DIR))
