import tensorflow as tf


def align_pca(X):
    n_pc_points = X.shape[1]
    C = tf.matmul(tf.transpose(X,[0, 2, 1]), X)
    s_v, u_v,v_v = tf.svd(C)
    v_v = tf.einsum("aij->aji", v_v)
    R_opt = tf.einsum("aij,ajk->aik", u_v, v_v)
    concat_R_opt = tf.tile(tf.expand_dims(R_opt,1), [1, n_pc_points, 1, 1])
    opt_labels =  tf.einsum("abki,abi->abk", concat_R_opt, X)
    return opt_labels

def align(X, Y):
    # align shapes from X to optimal tranformation between X and Y
    n_pc_points = X.shape[1]
    centered_y = tf.expand_dims(Y, 2)
    centered_x = tf.expand_dims(X, 2)
    # transpose y
    centered_y = tf.einsum('ijkl->ijlk', centered_y)
    mult_xy = tf.einsum('abij,abjk->abik', centered_y, centered_x)
    # sum
    C = tf.einsum('abij->aij', mult_xy)
    s, u,v = tf.svd(C)
    v = tf.einsum("aij->aji", v)
    R_opt = tf.einsum("aij,ajk->aik", u, v)
    concat_R_opt = tf.tile(tf.expand_dims(R_opt,1), [1, n_pc_points, 1, 1])
    opt_labels =  tf.einsum("abki,abi->abk", concat_R_opt, X)
    return opt_labels
