import os
import sys
import trimesh
import tensorflow as tf
from sklearn.neighbors import KDTree
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'train_logmap'))
import delaunay_tf
import tf_dataset
import pointnet_seg as model
BATCH_SIZE = 128
os.environ["CUDA_VISIBLE_DEVICES"]="0"
RESIZE=True
def safe_norm(x, epsilon=1e-8, axis=None):
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x) , axis=axis), epsilon))

def init_config():
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    return config


def resize(neighbor_points0):
    diag =  safe_norm(tf.reduce_max(neighbor_points0, axis = 1) - tf.reduce_min(neighbor_points0, axis = 1), axis = -1)
    diag = tf.tile(diag[:,tf.newaxis,tf.newaxis], [1, neighbor_points0.shape[1], neighbor_points0.shape[2]])
    neighbor_points0 = tf.divide(neighbor_points0, diag)
    return neighbor_points0

def geodesic_patch(neighbor_coordinates0,neighbors_index0, is_training):
    center_index = neighbors_index0[:,:1]
    center_coordinates = neighbor_coordinates0[:,:1]
    with tf.variable_scope("learn_geo_dist"):
        geo_distances = model.classifier(neighbor_coordinates0, is_training,  batch_size=BATCH_SIZE, activation=tf.nn.relu)
    geo_distances = tf.squeeze(tf.math.sigmoid(geo_distances), axis = 2)
    closests = tf.math.top_k(geo_distances[:,1:], k=n_nearest_neighbors)[1]
    neighbor_coordinates = tf.gather(neighbor_coordinates0[:,1:], closests, batch_dims=1)
    neighbors_index  = tf.gather(neighbors_index0[:,1:], closests, batch_dims=1)
    neighbor_coordinates = tf.concat([center_coordinates, neighbor_coordinates], axis =1)
    neighbors_index  = tf.concat([center_index,neighbors_index], axis =1)
    return neighbor_coordinates,  neighbors_index


def retrieve_triangles(shape_indices, shape_probs):
    trigs = [shape_indices[i][shape_probs[i]>0.5] for i in range(len(shape_indices))]
    trigs = np.array([ np.pad(trigs[i][:N_TRIG_NEIGHBORS], ((0,N_TRIG_NEIGHBORS - len(trigs[i])),(0,0)), constant_values=-1) for i in range(len(shape_indices))])
    return trigs

N_NEIGHBORS = 8
N_TRIG_NEIGHBORS = 10
def init_graph(X3D, n_neighbors,classifier_model, logmap_model):
    n_points = X3D.shape[0]

    config = init_config()
    current_type = tf.float32
    with tf.device('/gpu:'+str(0)):
        coord_3D = tf.constant(X3D,  dtype=tf.float32)
        points_neighbors0 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, n_neighbors+1])
        first_index_pl = tf.placeholder(tf.int32, shape=[BATCH_SIZE,])
        is_training = tf.constant(False, dtype = tf.bool)
        neighbor_points0 = tf.gather(coord_3D, points_neighbors0)
        neighbor_points0 = neighbor_points0 - tf.tile(neighbor_points0[:,:1],[1, n_neighbors+1, 1])

        if RESIZE:
            neighbor_points0 = resize(neighbor_points0)
        neighbor_points,  points_neighbors =  geodesic_patch(neighbor_points0,points_neighbors0, is_training)
        with tf.variable_scope("learn_logmap"):
            map = model.logmap(neighbor_points, is_training,  batch_size=BATCH_SIZE,activation=tf.nn.relu)
        predicted_map = tf.concat([ map, tf.zeros([ map.shape[0],  map.shape[1], 1])], axis = 2)

        target_triangles, target_indices = delaunay_tf.get_triangles_geo_batches(n_neighbors=n_nearest_neighbors,
                                                                                    gdist = predicted_map,
                                                                                    gdist_neighbors =points_neighbors[:,1:],
                                                                                    first_index =first_index_pl)

        logmap_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='learn_logmap')
        saver_logmap= tf.train.Saver(var_list = logmap_var)
        classifier_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='learn_geo_dist')
        saver_classifier= tf.train.Saver(var_list = classifier_var)

        init = tf.global_variables_initializer()

    session = tf.Session(config=config)
    session.run(init)

    saver = tf.train.Saver()

    saver_logmap.restore(session,logmap_model)
    saver_classifier.restore(session,classifier_model)


    ops = {"points_neighbors0" :points_neighbors0 ,
        "predicted_neighborhood_indices":points_neighbors,
        "first_index_pl" :first_index_pl ,
        "target_triangles" :target_triangles ,
        "target_indices" :target_indices ,
        "predicted_map": predicted_map}
    return session, ops

def reconstruct(name,classifier_model, logmap_model, in_path, res_path):
    logmap_points = np.loadtxt(os.path.join(in_path, name))
    name = name.replace('.xyz', "")
    X3D = logmap_points[:,:3]
    tree = KDTree(X3D)
    X3D_normals = np.zeros([X3D.shape[0],3])
    X3D_normals[:,2] = 1
    n_points = len(X3D)
    session,ops= init_graph(X3D, n_neighbors,classifier_model, logmap_model)
    points_indices =list(range(n_points))
    predicted_map = []
    triangles = []
    predicted_neighborhood_indices = []
    indices = []
    for step in range(1 + len(X3D)//BATCH_SIZE):
        if step%10==0:
            print("step: {}/{}".format(step,n_points//BATCH_SIZE))
        if (step+1)*BATCH_SIZE>n_points:
            current_points = points_indices[-BATCH_SIZE:]
        else:
            current_points = points_indices[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
        center_points = np.array(X3D[current_points])
        points_neighbors =tree.query(center_points, n_neighbors+1)[1]
        feed_dict = {ops['points_neighbors0']: points_neighbors,
                     ops['first_index_pl'] : current_points}
        to_run =   [ops["predicted_map"],ops['target_triangles'],ops['target_indices'], ops['predicted_neighborhood_indices']]
        res =session.run(to_run, feed_dict = feed_dict)
        if (step+1)*BATCH_SIZE>n_points:
            res[0] = res[0][-n_points%BATCH_SIZE:]
            res[1] =res[1][-n_points%BATCH_SIZE:]
            res[2] =res[2][-n_points%BATCH_SIZE:]
            res[3] =res[3][-n_points%BATCH_SIZE:]

        predicted_map.append(res[0])
        triangles.append(res[1])
        indices.append(res[2])
        predicted_neighborhood_indices.append(res[3])

    predicted_map = np.concatenate(predicted_map)
    triangles = np.concatenate(triangles)
    indices = np.concatenate(indices)
    predicted_neighborhood_indices = np.concatenate(predicted_neighborhood_indices)


    np.save(os.path.join(res_path,"predicted_map_{}.npy".format(name)), predicted_map)
    np.save(os.path.join(res_path,"predicted_neighborhood_indices_{}.npy".format(name)), predicted_neighborhood_indices)
    trimesh.Trimesh(X3D, indices[triangles>0.5]).export(os.path.join(res_path, "predicted_raw_mesh_{}.ply".format(name)))


if __name__ == '__main__':
    in_path = os.path.join(ROOT_DIR, 'data/test_data')
    res_path = os.path.join(ROOT_DIR, 'data/test_data/raw_prediction')
    if not os.path.exists(res_path): os.mkdir(res_path)
    # logmap_model = "log/log_thingifamous_logmap/model.ckpt"
    # classifier_model = "log/log_thingifamous_classifier/model.ckpt"
    logmap_model = os.path.join(ROOT_DIR, 'data/pretrained_models/pretrained_logmap/model.ckpt')
    classifier_model = os.path.join(ROOT_DIR, 'data/pretrained_models/pretrained_classifier/model.ckpt')
    n_neighbors = 120
    n_nearest_neighbors = 30

    # we evaluate all .xyz files in the in_path directory
    files = os.listdir(in_path)
    files = [x for x in files if x.endswith('.xyz')]

    for name in files:
        reconstruct(name, classifier_model, logmap_model, in_path, res_path)
