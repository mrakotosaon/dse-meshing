import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import trimesh
import open3d as o3d
from sklearn.neighbors import KDTree
import networkx as nx
import networkit as nk
import random
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
import numpy as np
import igraph
import gdist
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
import polyscope as ps
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def sample(shape, n_points):
    print('sampling {} vertices evenly'.format(n_points))
    with suppress_stdout_stderr():
        X3D, X3D_face_idx = trimesh.sample.sample_surface_even(shape, n_points)
        i=1000
        while len(X3D)<n_points:
            print("try sample again", len(X3D))
            X3D, X3D_face_idx = trimesh.sample.sample_surface_even(shape, n_points+i)
            i+=1000
            if len(X3D)>n_points:
                idx = np.random.choice(len(X3D), n_points, replace=False)
                X3D = X3D[idx]
                X3D_face_idx = X3D_face_idx[idx]
    return X3D, X3D_face_idx


def load_geo_data(filename, n_points, euclidian_neighbors):
    print('loading geodesic data for ', filename)
    data = np.loadtxt(filename)
    print('file loaded')
    geo_data = data.reshape([n_points, -1, 3])
    # replace the global indices of the original mesh by local ones
    geo_data[:,:,0] = euclidian_neighbors
    return geo_data

def write_geo_data(mesh_file, chosen_vertex_file, output_file, neighbors_file):
    run_command = "logmap/build/./bin/gc_project {} {} {} {}"
    run_command = run_command.format(mesh_file, chosen_vertex_file, output_file, neighbors_file)
    result = os.system(run_command)
    print('logmap c++ run status:', result)
    return result

def compute_logmap(data, n_logmap_points, n_logmap_neighbours,n_sampled_logmap_points, name):
    # sample logmap points
    X3D, _ = sample(data, n_logmap_points)
    original_mesh_tree = KDTree(data.vertices)
    original_mesh_mapping = original_mesh_tree.query(X3D)[1].reshape(-1)
    if len(set(original_mesh_mapping))<10000:
        remain =  list(set(list(range(len(data.vertices))))-set(original_mesh_mapping) )
        random.shuffle(remain)
        remain = remain[:10000 - len(set(original_mesh_mapping))]
        original_mesh_mapping =np.array(list(set(original_mesh_mapping)) + remain)


    X3D = data.vertices[original_mesh_mapping]
    logmap_points_tree = KDTree(X3D)
    dist_pc, ind_pc = logmap_points_tree.query(X3D, k=n_logmap_neighbours)

    # find correspondence in original mesh
    mesh_file = 'tmp_files/tmp_mesh_{}.obj'.format(name)
    output_file = 'tmp_files/computed_geo_{}.txt'.format(name)
    chosen_vertex_file = "tmp_files/vertices_{}.txt".format(name)
    neighbors_file = "tmp_files/neighbors_file_{}.txt".format(name)
    data.export(mesh_file)
    sampled_points = np.random.choice(n_logmap_points, n_sampled_logmap_points, replace=False)
    np.savetxt(chosen_vertex_file, original_mesh_mapping[sampled_points],fmt='%i')
    np.savetxt(neighbors_file, original_mesh_mapping[ind_pc[sampled_points]],fmt='%i')
    res = write_geo_data(mesh_file, chosen_vertex_file, output_file, neighbors_file)
    gdists = load_geo_data(output_file, n_sampled_logmap_points, ind_pc[sampled_points])
    os.remove(mesh_file)
    os.remove(output_file)
    os.remove(chosen_vertex_file)
    os.remove(neighbors_file)

    return X3D, gdists, res, sampled_points



def load_and_sample(mesh_filename,  n_logmap_points, n_logmap_neighbours, n_sampled_logmap_points):
    #data = [trimesh.load(os.path.join(root, fn), process=False) for fn in fns]
    #mesh = o3d.io.read_triangle_mesh(mesh_filename)
    #mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=True )

    #if mesh.body_count>1:
    #    mesh = mesh.split()[0]
    mesh = trimesh.load(mesh_filename, process=True)
    mesh.merge_vertices()
    n_vertices = len(mesh.vertices)
    mesh.vertices = mesh.vertices -np.mean(mesh.vertices,0)
    mesh.vertices/=mesh.scale
    # resize mesh
    print("subdividing mesh:",mesh_filename)

    while len(mesh.vertices)<100000:
        mesh = mesh.subdivide()
    print(len(mesh.vertices), "vertices")
    name = mesh_filename.split("/")[1].replace(".ply", "")
    logmap_points, logmap_data, res, sampled_logmap_points = compute_logmap(mesh, n_logmap_points, n_logmap_neighbours,n_sampled_logmap_points, name)
    return logmap_points, logmap_data, res, sampled_logmap_points

def convert_to(patch_data, name, root):
  """Converts a dataset to tfrecords."""
  filename = os.path.join(root, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(len(patch_data)):
      feature = {
        'patches': tf.train.Feature(float_list=tf.train.FloatList(value=patch_data[index].flatten())),
        }
      example = tf.train.Example(features=tf.train.Features(feature=feature))
      serialized = example.SerializeToString()
      writer.write(serialized)
  writer.close()


def prepare_patches(logmap_points,logmap_neighbors,logmap_maps, sampled_logmap_points):
    patches = []
    for i, value in enumerate(sampled_logmap_points):
        logmap_patch_points = logmap_points[logmap_neighbors[i]] - logmap_points[value]
        logmap_patch_map = logmap_maps[i]
        patch_info = np.concatenate([logmap_patch_points,logmap_patch_map], axis = 1)
        patches.append(patch_info)
    patches = np.array(patches)
    return patches

#######################################################################################################
def main(mesh_dataset_path, output_dataset_path,  n_logmap_points, n_logmap_neighbours, n_sampled_logmap_points):
    fns = os.listdir(mesh_dataset_path)
    fns = [x for x in fns if x.endswith(".ply")]

    n_shape = 0
    valid_shapes = []
    start = 0
    n_shape = len(valid_shapes)
    for fn in fns[start:]:
        logmap_points, logmap_data, res,sampled_logmap_points = load_and_sample(os.path.join(mesh_dataset_path, fn), n_logmap_points, n_logmap_neighbours, n_sampled_logmap_points)
        logmap_neighbors = logmap_data[:,:,0].astype(int)
        logmap_maps = logmap_data[:,:,1:]
        patches = prepare_patches(logmap_points,logmap_neighbors,logmap_maps,sampled_logmap_points)
        convert_to(patches, "logmap_patches_{}".format(n_shape), output_dataset_path)
        np.save(os.path.join(output_dataset_path,"patches_{}.npy".format(n_shape)), patches)
        np.save(os.path.join(output_dataset_path,"sampled_points_{}.npy".format(n_shape)), sampled_logmap_points)

        np.save(os.path.join(output_dataset_path,"logmap_points_{}.npy".format(n_shape)), logmap_points)
        if res==0:
            valid_shapes.append(fn)
            n_shape+=1
        text_file = open(os.path.join(output_dataset_path, "valid_files_list.txt"), "w")
        text_file.write("\n".join(valid_shapes))
        text_file.close()

if __name__ == '__main__':
    data_path = 'data'
    output_path = 'data/dataset'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists("tmp_files"):
        os.makedirs("tmp_files")

    n_logmap_points = 10000
    n_logmap_neighbours = 200
    n_sampled_logmap_points =  1000
    main(data_path, output_path, n_logmap_points, n_logmap_neighbours, n_sampled_logmap_points)
