import os
import sys
import csv
import math

import numpy as np
from numpy.core.defchararray import count, split


def open_write_path(file_path, open_mode):
    dir_path = os.path.dirname(file_path)
    if not (os.path.exists(dir_path) or dir_path == ''):
        os.makedirs(os.path.dirname(file_path))
    return open(file_path, open_mode)

def count_items(array, keys):
    n_keys = len(keys)
    return (np.tile(array, (n_keys, 1)) == np.expand_dims(keys, axis=-1)).sum(axis=-1)

def get_entropy(labels):
    assert isinstance(labels, np.ndarray)
    vals = np.unique(labels)
    val_count = count_items(labels, vals)
    total = len(labels)
    entropy = 0
    for count in val_count:
        entropy -= (count/total) * np.math.log2(count/total)
    return entropy

def get_cond_entropy(feature, labels):
    assert isinstance(feature, np.ndarray)
    assert isinstance(labels, np.ndarray)
    total_count = len(labels)
    feat_vals = np.unique(feature)
    feat_count = count_items(feature, feat_vals)
    cond_entropy = 0
    for f_val, f_count in zip(feat_vals.tolist(), feat_count):
        cond_entropy += (f_count/total_count) * get_entropy(labels[f_val == feature])
    return cond_entropy

def get_mi(feature, labels):
    assert isinstance(feature, np.ndarray)
    assert isinstance(labels, np.ndarray)
    return get_entropy(labels) - get_cond_entropy(feature, labels)

class Dataset:
    def __init__(self, header, data):
        self._header = header
        self._data = data

    @property
    def header(self):
        return self._header

    @property
    def data(self):
        return self._data

def load_data(file):
    with open(file, 'r') as in_file:
        tsv_reader = csv.reader(in_file, delimiter='\t')
        header = next(tsv_reader)
        data = np.array([line for line in tsv_reader])
    return Dataset(header, data)

def open_write_path(file_path, open_mode):
    dir_path = os.path.dirname(file_path)
    if not (os.path.exists(dir_path) or dir_path == ''):
        os.makedirs(os.path.dirname(file_path))
    return open(file_path, open_mode)

class DecisionTreeNode:

    def __init__(self, max_depth, train):
        self._children = {}
        self._split_attr_idx = None
        self._split_attr = None
        self._max_depth = max_depth

        # add a new node if max-depth is not reached
        if max_depth > 0:
            num_feat = train.data.shape[-1] - 1
            label_ent = get_entropy(train.data[:, num_feat])
            feat_cond_ent = [get_cond_entropy(train.data[:, i], train.data[:, num_feat])
                            for i in range(num_feat)]
            feat_mi = label_ent - feat_cond_ent
            max_mi_feat = np.argmax(feat_mi)
            # create an additional node if any feature has non-zero MI
            if feat_mi[max_mi_feat] > 0:
                self._split_attr_idx = max_mi_feat 
                self._split_attr = train.header[max_mi_feat]
                feat_vals = np.unique(train.data[:, max_mi_feat])
                for f_val in feat_vals:
                    mask = train.data[:, max_mi_feat] == f_val
                    self._children[f_val] = DecisionTreeNode(max_depth-1,
                                                             Dataset(train.header, train.data[mask]))
        # create data for majority voting
        self._labels = np.sort(np.unique(train.data[:, -1]))[::-1]
        self._label_counts = count_items(train.data[:, -1], self._labels)
    
    def get_depth(self):
        if self._split_attr_idx is None:
            return 0
        else:
            return 1 + max([child.get_depth() for child in self._children.values()])

    def predict(self, sample):
        if self._split_attr_idx is None:
            return self._labels[np.argmax(self._label_counts)]
        else:
            # labels are saved in reverse lexicographical order
            # On the event of tie, the label which comes first in this order is returned
            return self._children[sample[self._split_attr_idx]].predict(sample)


def train_tree(train_input, max_depth):
    train_data = load_data(train_input)
    return DecisionTreeNode(max_depth, train_data)


def generate_metrics(tree, train_file, test_file, train_out, test_out, metrics_out):
    metrics_file = open_write_path(metrics_out, 'w')
    for tag, in_f, out_f in [('train', train_file, train_out), ('test', test_file, test_out)]:
        error = 0
        total = 0
        with open(in_f, 'r') as in_file:
            tsv_reader = csv.reader(in_file, delimiter='\t')
            out_file = open_write_path(out_f, 'w')
            _ = next(tsv_reader)
            for line in tsv_reader:
                total += 1
                predict = tree.predict(line[:-1])
                if predict != line[-1]:
                    error += 1
                out_file.write(predict + '\n')
            out_file.close()
        metrics_file.write('error(%s): %0.12f\n'%(tag, error/total))
    metrics_file.close()


def pretty_print(tree):
    def print_node(node, max_depth, attr=None, val=None):
        if attr is not None:
            spacer = ''.join(['| ' for _ in range(max_depth - node._max_depth)])
            print(spacer + '%s = %s:'%(attr, val), end=' ')
        print_str = ['%d %s'%(count, label)
                        for count, label in zip(node._label_counts, node._labels)]
        print('['+'/'.join(print_str)+']')
        for child_value, child_node in node._children.items():
            print_node(child_node, max_depth, node._split_attr, child_value)

    print_node(tree, tree._max_depth)


if __name__=='__main__':
    assert len(sys.argv) == 7
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

    decision_tree = train_tree(train_input, max_depth)
    generate_metrics(decision_tree, train_input, test_input, train_out, test_out, metrics_out)
    pretty_print(decision_tree)
