import tensorflow as tf
import numpy as np
import config

def get_couples_matrix_sparse(shape):
    couples = []
    for i in range(1,shape):
        for j in range(i):
            couples.append([i,j])
    couples = np.array(couples)
    return couples

def safe_norm(x, epsilon=config.EPS, axis=None):
    return tf.sqrt(tf.maximum(tf.reduce_sum(x ** 2, axis=axis), epsilon))


def get_middle_points(center_point, B):
    center_point = tf.tile(center_point[:, tf.newaxis], [1, B.shape[1], 1])
    return (center_point + B)/2.0


def comp_half_planes(nn_coord,center_point):
    # compute the equations of the half planes
    n_points = nn_coord.shape[0]
    n_neighbors = nn_coord.shape[1]
    middle_points = get_middle_points(center_point, nn_coord)
    #this i snot in the same direction as 2D version?
    dir_vec=  nn_coord - center_point[:,tf.newaxis,:]
    half_planes_normal =  tf.divide(dir_vec,tf.maximum(tf.tile(safe_norm(dir_vec, axis = -1)[:,:,tf.newaxis],[1,1, 2]),config.EPS))
    col3 = - (middle_points[:,:,0]*half_planes_normal[:,:,0] + middle_points[:,:,1]*half_planes_normal[:,:,1] )
    half_planes = tf.concat([half_planes_normal, col3[:,:,tf.newaxis]], axis=-1)
    return half_planes

def get_is_trig_exact(inter_dist, n_neighbors):
    n_points = inter_dist.shape[0]
    inter_dist = -tf.sign(inter_dist)
    is_triangle = tf.reduce_sum(inter_dist, axis = 2)
    is_triangle = tf.where(is_triangle<n_neighbors, tf.zeros_like(is_triangle), tf.ones_like(is_triangle))
    return is_triangle


def compute_intersections(half_planes, couples):
    # compute the intersections between the couples of half planes
    inter = tf.linalg.cross(tf.gather(half_planes,couples[:,0], axis=1), tf.gather(half_planes,couples[:,1], axis=1))
    mask = tf.abs(inter[:,:,2])<config.EPS
    inter = tf.divide(inter,tf.tile(tf.expand_dims(tf.where(mask, tf.ones_like(inter[:,:,2]),inter[:,:,2]), 2),[1, 1,3])) # what to do if no intersection?
    inter = tf.where(tf.tile(mask[:,:,tf.newaxis], [1, 1, 3]), tf.ones_like(inter)*10.0,inter)# if no intersection put point far away
    return inter



def compute_triangles_local_geodesic_distances(nn_coord, center_point, couples):
    n_neighbors = nn_coord.shape[1].value
    n_trigs = couples.shape[0]
    nn_coord = nn_coord[:,:,:2]
    center_point = center_point[:,:2]
    half_planes =  comp_half_planes(nn_coord, center_point)
    intersections= compute_intersections(half_planes, couples)
    intersection_couples = tf.tile(couples[tf.newaxis,:,:],[center_point.shape[0], 1, 1])
    # compute the distance between the intersection points (N**2 points) and the half planes (N)
    inter_dist0 = tf.reduce_sum(tf.multiply(tf.tile(half_planes[:,tf.newaxis,:,:],[1, n_trigs, 1, 1]) ,tf.tile(intersections[:,:,tf.newaxis,:],[1,1,n_neighbors, 1])), axis=-1)
    index_couples_a = tf.tile(tf.range(center_point.shape[0])[:,tf.newaxis,tf.newaxis], [1, n_trigs, 2])
    index_couples_b = tf.tile(tf.range(n_trigs)[tf.newaxis,:,tf.newaxis], [center_point.shape[0], 1, 2])
    index_couples = tf.stack([index_couples_a, index_couples_b, intersection_couples], axis = -1)
    to_ignore = tf.scatter_nd(tf.reshape(index_couples, [-1, 3]), tf.ones([center_point.shape[0]*n_trigs*2]), inter_dist0.shape)
    inter_dist0 = tf.where(to_ignore>0.5, -tf.ones_like(inter_dist0)*1e10,inter_dist0)
    inter_dist = tf.where(tf.abs(inter_dist0)<config.EPS,-tf.ones_like(inter_dist0)*1e10, inter_dist0)
    is_triangle_exact = get_is_trig_exact(inter_dist, n_neighbors)
    return is_triangle_exact,  intersection_couples

def get_triangles_geo_batches(  n_neighbors=60, gdist=None, gdist_neighbors=None, first_index=None):
    couples = tf.constant(get_couples_matrix_sparse(n_neighbors), dtype = tf.int32)
    nn_coord = gdist[:,1:]
    center_point = gdist[:,0]
    exact_triangles, local_indices= compute_triangles_local_geodesic_distances(nn_coord,  center_point, couples)
    global_indices = tf.gather(gdist_neighbors, local_indices, batch_dims=1)
    first_index = tf.tile(first_index[:,tf.newaxis,tf.newaxis],[1, global_indices.shape[1], 1])
    global_indices = tf.concat([first_index, global_indices], axis = 2)
    return exact_triangles, global_indices
