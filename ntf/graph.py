import tensorflow as tf
from google.protobuf import text_format


__all__ = ['save_graph_pb', 'save_graph_pbtxt', 'save_graph_tb',
           'load_graph_pb', 'load_graph_pbtxt']


def save_graph_pbtxt(fpath, graph=None):
    if fpath is None:
        return
    if graph is None:
        graph = tf.get_default_graph()

    tf.io.write_graph(graph.as_graph_def(), '.', fpath)


def save_graph_pb(fpath, graph=None):
    if fpath is None:
        return
    if graph is None:
        graph = tf.get_default_graph()

    with tf.gfile.GFile(fpath, "wb") as f:
        f.write(graph.as_graph_def().SerializeToString())


def load_graph_pbtxt(fpath, input_map=None, return_elements=None):
    with tf.gfile.GFile(fpath, 'r') as f:
        graph_def = tf.GraphDef()
        text_format.Merge(f.read(), graph_def)
        return tf.import_graph_def(graph_def, name='',
                                   input_map=input_map,
                                   return_elements=return_elements)


def load_graph_pb(fpath, input_map=None, return_elements=None):
    with tf.gfile.GFile(fpath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        return tf.import_graph_def(graph_def, name='',
                                   input_map=input_map,
                                   return_elements=return_elements)


def save_graph_tb(fpath, graph=None):
    if graph is None:
        graph = tf.get_default_graph()
    writer = tf.summary.FileWriter(fpath, graph=graph)
    writer.add_graph(graph)
