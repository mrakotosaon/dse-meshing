import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
import numpy as np
from sklearn.cluster import DBSCAN
import time
import functools
import multiprocessing
from multiprocessing import RawArray


def ICP(X, Y):
    n_pc_points = X.shape[1]
    mu_x = np.mean(X, axis = 1)
    mu_y =  np.mean(Y, axis = 1)
    concat_mu_y = np.tile(np.expand_dims(mu_y,1), [1, n_pc_points, 1])
    concat_mu_x = np.tile(np.expand_dims(mu_x,1), [1, n_pc_points, 1])
    centered_y = np.expand_dims(Y - concat_mu_y, 2)
    centered_x = np.expand_dims(X - concat_mu_x, 2)
    # transpose y
    centered_y = np.einsum('ijkl->ijlk', centered_y)

    mult_xy = np.einsum('abij,abjk->abik', centered_y, centered_x)
    # sum
    C = np.einsum('abij->aij', mult_xy)
    u,s,v = np.linalg.svd(C)
    v = np.einsum("aij->aji", v)

    R_opt = np.einsum("aij,ajk->aik", u, v)
    t_opt = mu_y - np.einsum("aki,ai->ak", R_opt, mu_x)
    concat_R_opt = np.tile(np.expand_dims(R_opt,1), [1, n_pc_points, 1, 1])
    concat_t_opt = np.tile(np.expand_dims(t_opt,1), [1, n_pc_points, 1])
    opt_labels =  np.einsum("abki,abi->abk", concat_R_opt, X) + concat_t_opt
    return opt_labels


def align_current_neighborhood(center_map_indices,neighbor_map_indices,center_map,neighbor_map):
    intersection = list(set(center_map_indices).intersection(set(neighbor_map_indices)))
    aligned_neighborhood = None
    target_idx = None
    if len(intersection)>10:
        target_idx = np.squeeze(np.array([np.where(center_map_indices==i) for i in intersection]))
        source_idx = np.squeeze(np.array([np.where(neighbor_map_indices==i) for i in intersection]))
        source = neighbor_map[source_idx]
        target = center_map[target_idx]
        aligned_neighborhood = np.squeeze(ICP(np.array([source]), np.array([target])))
    return aligned_neighborhood, target_idx



def correct_point(points, appearance, point_weights):
    if appearance>0:
        corrected_point = np.zeros([3])
        epsilons = [0.6*0.05,0.6*0.12, 0.6*0.15, 0.6*0.2]
        cleaned_points = []
        attempt_number = 0
        while len(cleaned_points)<=0 and attempt_number<3:
            clustering = DBSCAN(eps=epsilons[attempt_number], min_samples=5).fit(points, sample_weight=point_weights)
            cleaned_points = points[clustering.labels_==0]
            tmp_point_weights = point_weights[clustering.labels_==0]
            attempt_number+=1
        if len(cleaned_points)>0:
            corrected_point[:2] = np.average(cleaned_points[:,:2], axis=0, weights=tmp_point_weights)
        else:
            corrected_point = np.array([0.5,0.5, 0.0])
    else:
        corrected_point = np.array([0.5,0.5, 0.0])

    return corrected_point

def align_patch(predicted_map, predicted_neighborhood_indices, center_point):
    center_map = predicted_map[center_point]
    center_map_indices = predicted_neighborhood_indices[center_point]
    center_neighbors = predicted_neighborhood_indices[center_point][1:]
    neighbors_maps = predicted_map[center_neighbors]
    neighbors_maps_indices = predicted_neighborhood_indices[center_neighbors]
    aligned_neighborhoods = np.zeros([len(center_neighbors), len(center_map), 3])
    aligned_neighborhood_weight = np.zeros([len(center_neighbors), len(center_map)])
    for neighbor in range(n_nearest_neighbors):
        aligned_neighborhood,  target_idx = align_current_neighborhood(center_map_indices, neighbors_maps_indices[neighbor],center_map,neighbors_maps[neighbor])
        if aligned_neighborhood is not None:
            aligned_neighborhood_weight[neighbor, target_idx] = 1
            aligned_neighborhoods[neighbor, target_idx] = aligned_neighborhood

    aligned_neighborhoods =  np.concatenate([center_map[np.newaxis,:,:], aligned_neighborhoods], axis = 0)
    aligned_neighborhood_weight = np.concatenate([np.ones([1, len(center_map)]), aligned_neighborhood_weight], axis = 0)


    distance_to_center_weight = np.maximum(0.2,np.power((1-np.linalg.norm(center_map,axis=1)),2))
    distance_to_center_weight = np.tile(distance_to_center_weight[:,np.newaxis],[1, n_nearest_neighbors+1])
    aligned_neighborhood_weight = np.multiply(aligned_neighborhood_weight, distance_to_center_weight)
    distance_to_neighbor_center = np.maximum(0.2,np.power(1-np.linalg.norm(neighbors_maps, axis =2), 2))
    aligned_neighborhood_weight[1:] = np.multiply(distance_to_neighbor_center,aligned_neighborhood_weight[1:])

    n_appearance = np.sum(aligned_neighborhood_weight, axis =0)
    corrected_patch = np.zeros([n_nearest_neighbors+1, 3])


    for point in range(n_nearest_neighbors+1):
        points = aligned_neighborhoods[:,point][aligned_neighborhood_weight[:,point]>0.1]
        point_weights = aligned_neighborhood_weight[:,point][aligned_neighborhood_weight[:,point]>0.1]

        corrected_point = correct_point(points, n_appearance[point], point_weights)
        corrected_patch[point] = corrected_point


    return corrected_patch

def main(file):
    predicted_map = np.load(os.path.join(raw_predictions,"predicted_map_{}.npy".format(file)))

    points = np.loadtxt(os.path.join(in_path,"{}.xyz".format(file)))
    corrected_maps = np.zeros_like(predicted_map)
    n_points = len(predicted_map)
    predicted_neighborhood_indices = np.load(os.path.join(raw_predictions,"predicted_neighborhood_indices_{}.npy".format(file)))
    full_errors = []
    BATCH_SIZE = 64#512#16
    shared_predicted_map = RawArray('d',10000*(n_nearest_neighbors+1)*3)
    shared_predicted_map = np.frombuffer(shared_predicted_map, dtype=np.float64).reshape(10000, (n_nearest_neighbors+1), 3)
    np.copyto(shared_predicted_map, predicted_map)

    shared_predicted_neighborhood_indices =RawArray('i',10000*(n_nearest_neighbors+1))
    shared_predicted_neighborhood_indices = np.frombuffer(shared_predicted_neighborhood_indices, dtype=np.int32).reshape(10000, (n_nearest_neighbors+1))
    np.copyto(shared_predicted_neighborhood_indices, predicted_neighborhood_indices)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    global align_patch_func
    def align_patch_func(shared_predicted_map,shared_predicted_neighborhood_indices, i):
        return  align_patch(shared_predicted_map, shared_predicted_neighborhood_indices, i)

    jobs = []
    start = time.time()
    print('start:', file)
    with multiprocessing.Pool(64) as pool:
        corrected_maps = pool.map(functools.partial(align_patch_func,shared_predicted_map,shared_predicted_neighborhood_indices), range(n_points))
        corrected_maps = np.array(corrected_maps)
        np.save(os.path.join(res_path,'corrected_maps_{}.npy'.format(file)), corrected_maps)
    end = time.time()
    #print("{:3.3f} seconds".format(end - start))

if __name__ == '__main__':
    print('patches alignment: this step took around 20 sec per shape of 10k points on our machine')
    in_path = os.path.join(ROOT_DIR, 'data/test_data')
    raw_predictions = os.path.join(ROOT_DIR, 'data/test_data/raw_prediction')
    res_path = os.path.join(ROOT_DIR, 'data/test_data/aligned_prediction')
    if not os.path.exists(res_path): os.mkdir(res_path)
    # we evaluate all .xyz files in the in_path directory
    files = os.listdir(in_path)
    files = [x.replace('.xyz', '') for x in files if x.endswith('.xyz')]
    n_nearest_neighbors = 30
    for file in files:
        main(file)
