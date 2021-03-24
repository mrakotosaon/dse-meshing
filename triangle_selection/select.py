import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
import numpy as np
import random
import pickle
import trimesh
BBOX = 1000.0 # we resize the shape to this bounding box to avoid numerical errors in the (external) postprocessing code

def write_candidates(file):
    points = np.loadtxt(os.path.join(in_path, "{}.xyz".format(file)))
    bbox_diag = np.linalg.norm(np.max(points, axis = 0)-np.min(points, axis = 0))
    points*=BBOX/bbox_diag
    trig_probs = np.load(os.path.join(align_path, "patch_frequency_count_{}.npy".format(file)))

    indices = np.reshape(trig_probs[:,:3], [-1, 3])
    labels = np.reshape(trig_probs[:,3:], [-1])
    points = points.astype(str)
    bins = np.zeros_like(labels)
    bins[labels==3]=1
    bins[labels==2]=2
    bins[labels==1]=3
    bins = np.concatenate([indices, bins[:,np.newaxis]], axis = 1).astype(str)
    content = str(len(points)) + '\n'
    content+= "\n".join([" ".join(list(x)) for x in points.astype(str)]) + '\n'
    content+= str(len(indices)) + '\n'
    content+= "\n".join([" ".join(list(x)) for x in bins])
    with open(os.path.join(res_path, "pred_{}.txt".format(file)), 'w') as f:
        f.write(content)
    return bbox_diag

if __name__ == '__main__':
    in_path = os.path.join(ROOT_DIR, 'data/test_data')
    align_path = os.path.join(ROOT_DIR, 'data/test_data/aligned_prediction')
    res_path = os.path.join(ROOT_DIR, 'data/test_data/select')
    if not os.path.exists(res_path): os.mkdir(res_path)
    files = os.listdir(in_path)
    files = [x.replace('.xyz', "") for x in files if x.endswith('.xyz')]
    bbox_diag = []
    for file in files:
        print("triangle selection:",file)
        bbox_diag.append(write_candidates(file))
        arg1 = os.path.join(res_path, "pred_{}.txt".format(file))
        arg2 = os.path.join(res_path, "final_mesh_{}.ply".format(file))
        os.system(os.path.join(BASE_DIR, "postprocess/build/postprocess {} {}".format(arg1, arg2)))
    for i, file in enumerate(files):
        print("triangle resize:",file)
        arg2 = os.path.join(res_path, "final_mesh_{}.ply".format(file))
        mesh = trimesh.load(arg2, process=False)
        mesh.vertices*=bbox_diag[i]/BBOX
        mesh.export(arg2)
