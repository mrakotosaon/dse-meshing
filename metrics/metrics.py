import numpy as np
from sklearn.neighbors import KDTree
import os
import sys
import trimesh
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)

def sample(mesh, n_points=100000):
    # attempt to sample points evenly 3 times if the 1rst time fails
    np.random.seed(0)
    samples = trimesh.sample.sample_surface_even(mesh, n_points)[0]
    i = 10000
    for n in range(3):
        if len(samples)<n_points:
            print('points sampled: {} we need to sample more points.'.format(len(samples)))
            samples = trimesh.sample.sample_surface_even(mesh, n_points + i*(n+1))[0]
        if len(samples)>n_points:
            idx = np.random.choice(len(samples), n_points, replace=False)
            samples = samples[idx]
    if len(samples)<n_points:
        print("we could not sample enough points evenly the sampling will be uneven")
        rest_samples = trimesh.sample.sample_surface(mesh, n_points-len(samples))[0]
        samples = np.concatenate([samples, rest_samples])
    return samples

def chamfer_distance(sA, sB):
    treeA = KDTree(sA)
    treeB = KDTree(sB)
    distB2A = treeA.query(sB)[0]
    distA2B = treeB.query(sA)[0]
    cd = np.mean(distA2B) + np.mean(distB2A)
    return cd

def resize_shapes(mesh, gt_mesh):
    diag = np.linalg.norm(-np.min(gt_mesh.vertices , axis = 0) + np.max(gt_mesh.vertices , axis = 0))
    bbox_center  = (np.min(gt_mesh.vertices , axis = 0) + np.max(gt_mesh.vertices , axis = 0))/2
    mesh.vertices  -= bbox_center
    mesh.vertices/=diag
    gt_mesh.vertices  -= bbox_center
    gt_mesh.vertices/=diag
    return mesh, gt_mesh

def evaluate_CD(mesh, gt_mesh):
    mesh_samples = sample(mesh)
    gt_samples = sample(gt_mesh)
    return  chamfer_distance(mesh_samples, gt_samples)

def non_manifoldness(mesh):
    mesh = trimesh.Trimesh(mesh.vertices, np.unique(np.sort(np.array(mesh.faces), axis = 1),axis=0))
    ordered_edges = [np.sort(x) for x in mesh.edges]
    uni,count = np.unique(ordered_edges, return_counts=True, axis=0)
    edge_manifoldness = len(np.argwhere(count==2))/float(len(count))
    return 1-edge_manifoldness

def adapted_normals(gt_mesh, gt_triangle_id, gt_closest):
    # we average the gt normals when the point is very close to the edge of the face (closer than 0.1% of the bbox diagonal)
    gt_triangles = gt_mesh.triangles[gt_triangle_id]
    gt_faces = gt_mesh.faces[gt_triangle_id]
    gt_closest_normals = np.array(gt_mesh.face_normals[gt_triangle_id])
    e1 = gt_triangles[:,1]- gt_triangles[:,0]
    e2 = gt_triangles[:,2]- gt_triangles[:,0]
    e3 = gt_triangles[:,2]- gt_triangles[:,1]
    e1/=np.linalg.norm(e1, axis = 1)[:,np.newaxis]
    e2/=np.linalg.norm(e2, axis = 1)[:,np.newaxis]
    e3/=np.linalg.norm(e3, axis = 1)[:,np.newaxis]
    v1 = gt_closest- gt_triangles[:,0]
    v2 = gt_closest- gt_triangles[:,0]
    v3 = gt_closest-  gt_triangles[:,1]
    dist_to_e1 = np.linalg.norm(v1 - np.sum(np.multiply(v1, e1), axis = 1)[:,np.newaxis]*e1, axis = 1)
    dist_to_e2 = np.linalg.norm(v2 - np.sum(np.multiply(v2, e2), axis = 1)[:,np.newaxis]*e2, axis = 1)
    dist_to_e3 = np.linalg.norm(v3 - np.sum(np.multiply(v3, e3), axis = 1)[:,np.newaxis]*e3, axis = 1)
    distances = np.stack([dist_to_e1,dist_to_e2,dist_to_e3], axis = 1)
    sorted_dist_idx = np.argsort(distances, axis = 1)
    sorted_dist = np.take_along_axis(distances, sorted_dist_idx, axis=1)
    to_correct = 3*1e-3>sorted_dist[:,0]
    edges = np.array([[0, 1], [0, 2], [1, 2]])
    count=0
    edges_dict = [['{}_{}'.format(e[0], e[1]), k] for k, e in enumerate(gt_mesh.edges_unique)]
    edges_dict+=[['{}_{}'.format(e[1], e[0]), k] for k, e in enumerate(gt_mesh.edges_unique)]
    edges_dict = dict(edges_dict)
    for k in range(len(gt_closest_normals)):
        if to_correct[k]:
            count+=1
            closest_edge =  edges[sorted_dist_idx[k][0]]
            close_edge1 = gt_faces[k][closest_edge[0]]
            close_edge2 = gt_faces[k][closest_edge[1]]
            edge_idx =edges_dict["{}_{}".format(close_edge1, close_edge2)]
            adjacent_faces = gt_mesh.face_adjacency[edge_idx]
            corrected_normal = gt_mesh.face_normals[adjacent_faces]
            corrected_normal = np.sum(corrected_normal, axis = 0)
            corrected_normal/=np.linalg.norm(corrected_normal)
            gt_closest_normals[k] = corrected_normal
    return gt_closest_normals

def normal_reconstruction(mesh, gt_mesh):
    mesh.fix_normals()
    gt_mesh.fix_normals()
    gt_closest_point, _, gt_triangle_id = trimesh.proximity.closest_point(gt_mesh, mesh.vertices)
    if len(gt_mesh.face_adjacency)==len(gt_mesh.edges_unique):
        # if the mesh is manifold we compute adapted normals
        gt_closest_normals = adapted_normals(gt_mesh,gt_triangle_id,gt_closest_point)
    else:
        print('the mesh is not manifold')
        gt_triangles = gt_mesh.triangles[gt_triangle_id]
        gt_closest_normals = np.array(gt_mesh.face_normals[gt_triangle_id])

    normal_angles = np.minimum(trimesh.geometry.vector_angle(np.stack([mesh.vertex_normals, gt_closest_normals], axis = 1)),
                               trimesh.geometry.vector_angle(np.stack([mesh.vertex_normals,-gt_closest_normals], axis = 1)))
    return np.rad2deg(np.mean(normal_angles))

if __name__ == '__main__':
    # please change the paths here to evaluate other shapes
    path = os.path.join(ROOT_DIR, 'data/test_data/select')
    gt_path = os.path.join(ROOT_DIR, 'data/test_data_famousthingi')
    cds = []
    nms = []
    NORMAL_RECONS=False # by default we do not compute normal reconstruction loss as it takes time
    normal_reconstruction_metric = []
    files = os.listdir(path)
    shapes = [f.split('final_mesh_')[1].replace('.ply','') for f in files if f.endswith('.ply')]

    for shape in shapes:
        print('ours')
        mesh = trimesh.load(os.path.join(path, "final_mesh_{}.ply".format(shape)))
        mesh.faces = np.unique(np.sort(mesh.faces, axis = 1), axis = 0)
        gt_mesh = trimesh.load(os.path.join(gt_path, "{}.ply".format(shape)))
        gt_mesh.merge_vertices()
        mesh, gt_mesh = resize_shapes(mesh, gt_mesh)
        gt_mesh.fix_normals()
        mesh.fix_normals()
        cds.append(evaluate_CD(mesh, gt_mesh))
        nms.append(non_manifoldness(mesh))
        print("shape: {:20s} CD: {:3.4f}*1e-2 NM: {:3.3f}%".format(shape, cds[-1]*100, nms[-1]*100))
        if NORMAL_RECONS:
            normal_reconstruction_metric.append(normal_reconstruction(mesh, gt_mesh))
            print("{}NR: {:3.3f}°".format(" "*28,normal_reconstruction_metric[-1]))

    print("average: CD: {:3.4f}*1e-2 NM: {:3.3f}%".format(np.mean(cds)*100, np.mean(nms)*100))
    if NORMAL_RECONS:
        print("NR {:3.3f}°".format(np.mean(normal_reconstruction_metric)))
