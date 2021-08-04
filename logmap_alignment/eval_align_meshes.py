import os
import sys
import trimesh
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
from sklearn.neighbors import KDTree
import delaunay_tf
BATCH_SIZE = 128

def init_config():
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    return config


def init_graph(X3D,X3D_normals, n_neighbors):
    n_points = X3D.shape[0]

    config = init_config()
    current_type = tf.float32
    with tf.device('/gpu:'+str(0)):
        normals = tf.Variable(tf.convert_to_tensor(X3D_normals,dtype=tf.float32), dtype=tf.float32)
        first_index_pl = tf.placeholder(tf.int32, shape=[BATCH_SIZE,])
        corrected_map = tf.placeholder(tf.float32, shape=[BATCH_SIZE, n_nearest_neighbors+1, 3])
        corrected_points_neighbors = tf.placeholder(tf.int32, shape=[BATCH_SIZE, n_nearest_neighbors+1])

        corrected_approx_triangles, corrected_indices = delaunay_tf.get_triangles_geo_batches(n_neighbors=n_nearest_neighbors,
                                                                                       gdist = corrected_map,
                                                                                       gdist_neighbors =corrected_points_neighbors[:,1:],
                                                                                       first_index =first_index_pl)



        init = tf.global_variables_initializer()

    session = tf.Session(config=config)
    session.run(init)

    saver = tf.train.Saver()

    ops = {"corrected_indices":corrected_indices,
            "corrected_approx_triangles":corrected_approx_triangles,
            "first_index_pl" :first_index_pl ,
            "corrected_map":corrected_map,
            "corrected_points_neighbors":corrected_points_neighbors,}
    return session, ops

def reconstruct(name):
    logmap_points = np.loadtxt(os.path.join(in_path, name))
    name = name.replace('.xyz', "")
    X3D = logmap_points
    tree = KDTree(logmap_points)
    X3D_normals = np.zeros([X3D.shape[0],3])
    X3D_normals[:,2] = 1

    n_points = len(logmap_points)
    session,ops= init_graph(X3D,X3D_normals, n_neighbors)
    points_indices =list(range(n_points))

    predicted_neighborhood_indices = np.load(os.path.join(raw_prediction_path,"predicted_neighborhood_indices_{}.npy".format(name)))#np.load("icp_consistency_results/predicted_neighborhood_indices.npy" )
    corrected_predicted_map = np.load(os.path.join(res_path, "corrected_maps_{}.npy".format(name)))

    triangles = []
    indices = []
    for step in range(1 + len(X3D)//BATCH_SIZE):
        if step%50==0:
            print("step: {}/{}".format(step,n_points//BATCH_SIZE))
        if (step+1)*BATCH_SIZE>n_points:
            current_points = points_indices[-BATCH_SIZE:]
        else:
            current_points = points_indices[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
        center_points = np.array(X3D[current_points])
        points_neighbors =tree.query(center_points, n_neighbors+1)[1]
        feed_dict = {ops['first_index_pl'] : current_points,
                     ops['corrected_map']:corrected_predicted_map[current_points],
                     ops['corrected_points_neighbors']:predicted_neighborhood_indices[current_points]}
        to_run = [ops['corrected_approx_triangles'],ops['corrected_indices']]
        res =session.run(to_run, feed_dict = feed_dict)
        if (step+1)*BATCH_SIZE>n_points:
            res[0] = res[0][-n_points%BATCH_SIZE:]
            res[1] =res[1][-n_points%BATCH_SIZE:]

        triangles.append(res[0])
        indices.append(res[1])
    indices = np.concatenate(indices)
    triangles = np.concatenate(triangles)

    trigs = np.sort(np.reshape(indices[triangles>0.5],[-1,3]), axis = 1)
    uni,inverse, count = np.unique(trigs, return_counts=True, axis=0, return_inverse=True)
    triangle_occurence = count[inverse]
    np.save(os.path.join(res_path, "patch_frequency_count_{}.npy".format(name)), np.concatenate([uni, count[:,np.newaxis]], axis = 1) )


if __name__ == '__main__':
    # evaluate the new meshes and count frequency of triangles
    in_path = os.path.join(ROOT_DIR, 'data/test_data')
    raw_prediction_path = os.path.join(ROOT_DIR, 'data/test_data/raw_prediction')
    res_path = os.path.join(ROOT_DIR, 'data/test_data/aligned_prediction')
    n_neighbors = 120
    n_nearest_neighbors = 30

    # we evaluate all .xyz files in the in_path directory
    files = os.listdir(in_path)
    files = [x for x in files if x.endswith('.xyz')]

    for name in files:
        reconstruct(name)
