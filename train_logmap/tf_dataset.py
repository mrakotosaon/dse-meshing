import tensorflow as tf

def parse_proto(example_proto):
    features = {
    'patches': tf.FixedLenFeature((N_NEIGHBORS*5,), tf.float32, default_value = tf.zeros([N_NEIGHBORS*5])),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    parsed_features['patches'] = tf.reshape(parsed_features['patches'],[N_NEIGHBORS, 5])
    return  parsed_features['patches']

def dataset(filenames, batch_size, n_patches = 1500, n_neighbors=200):
    global N_NEIGHBORS
    N_NEIGHBORS = n_neighbors
    buffer_size = n_patches*len(filenames)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_proto)
    dataset = dataset.shuffle(buffer_size,reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    #dataset = dataset.prefetch(buffer_size)
    iterator = dataset.make_one_shot_iterator()
    next = iterator.get_next()
    return iterator,  dataset

if __name__ == '__main__':
    filenames = ["../preprocessing/planes_{}.tfrecords".format(i) for i in [1,2,3]]
    iterator, dataset = dataset(filenames)
    next = iterator.get_next()
    with tf.Session() as sess:
        vals = sess.run(next)
        print(vals[1][0,0])
