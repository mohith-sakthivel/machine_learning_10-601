import os
import csv
import sys

import numpy as np
from numpy.core.defchararray import mod


class DataSet:
    """
    Object for storing label and feature data
    """

    def __init__(self, num_classes=None, feature_dim=None):
        self._labels = []
        self._features = []
        self._num_classes = num_classes
        self._feature_dim = feature_dim

    def __len__(self):
        return len(self._labels)

    def append(self, label, feat):
        self._labels.append(label)
        self._features.append(feat)

    @property
    def feature_dim(self):
        if self._feature_dim is None:
            self._feature_dim = len(self._features[0])
        return self._feature_dim

    @property
    def num_classes(self):
        if self._num_classes is None:
            return max(self._labels) + 1
        return self._num_classes

    def __iter__(self):
        return zip(self._labels, self._features)


def load_data(path, dataset_kwargs={}):
    """Load formatted input data from a file"""
    with open(path, 'r') as read_file:
        reader = csv.reader(read_file)
        data = DataSet(**dataset_kwargs)
        for line in reader:
            label = int(line[0])
            feature = line[1:]
            data.append(label, feature)
    return data


def open_write_path(file_path, open_mode):
    dir_path = os.path.dirname(file_path)
    if not (os.path.exists(dir_path) or dir_path == ''):
        os.makedirs(os.path.dirname(file_path))
    return open(file_path, open_mode)


D_TYPE = np.float32


class NN():

    def __init__(self, in_dim, hid_units, num_classes, lr, init_flag):
        if init_flag == '1':
            def make_fn(size): return np.random.uniform(
                size=size, low=-0.1, high=0.1).astype(D_TYPE)
        elif init_flag == '2':
            def make_fn(shape): return np.zeros(shape, dtype=D_TYPE)
        else:
            raise ValueError
        self.lr = lr
        self.alpha_w = make_fn((hid_units, in_dim))
        self.alpha_b = np.zeros(hid_units, dtype=D_TYPE)
        self.beta_w = make_fn((num_classes, hid_units))
        self.beta_b = np.zeros(num_classes, dtype=D_TYPE)

    def __call__(self, x):
        x = np.matmul(self.alpha_w, np.array(x, dtype=D_TYPE)) + self.alpha_b
        x = 1/(1 + np.exp(-x))
        x = np.matmul(self.beta_w, x) + self.beta_b
        x = np.exp(x)
        return x/x.sum(axis=-1)

    def update_weights(self, grads):
        for param, grad in grads.items():
            new_val = getattr(self, param) - self.lr * grad
            self.__setattr__(param, new_val)

    def sgd_step(self, x, y):
        # forward pass
        x = np.array(x, dtype=D_TYPE)
        a = np.matmul(self.alpha_w, x) + self.alpha_b
        z = 1/(1 + np.exp(-a))
        b = np.matmul(self.beta_w, z) + self.beta_b
        y_logits = np.exp(b)
        # backward pass
        grads = {}
        d_b = (y_logits)/y_logits.sum(axis=-1)
        d_b[y] -= 1
        grads['beta_b'] = d_b
        grads['beta_w'] = np.matmul(d_b.reshape(-1, 1), z.reshape(1, -1))
        d_z = np.matmul(d_b, self.beta_w)
        d_a = d_z * z * (1-z)
        grads['alpha_b'] = d_a
        grads['alpha_w'] = np.matmul(d_a.reshape(-1, 1), x.reshape(1, -1))
        self.update_weights(grads)


def mean_cross_entropy(model, data):
    ce = []
    for label, feat in data:
        probs = model(feat)
        ce.append(-np.log(probs[label]))
    return sum(ce)/len(ce)


def write_preds(model, data, out_path):
    with open_write_path(out_path, 'w') as out_file:
        error = 0
        for label, feat in data:
            probs = model(feat)
            pred = np.argmax(probs, axis=-1)
            out_file.write(f'{pred}\n')
            if pred != label:
                error += 1
    return error/len(data)


def train_model(train_data, valid_data, hid_units, lr, num_epoch, init_flag, metrics_out, train_out, valid_out):
    net = NN(train_data.feature_dim, hid_units,
             train_data.num_classes, lr, init_flag)
    out_file = open_write_path(metrics_out, 'w')
    for epoch in range(1, num_epoch+1):
        for label, feat in train_data:
            net.sgd_step(feat, label)
        ce_train = mean_cross_entropy(net, train_data)
        ce_valid = mean_cross_entropy(net, valid_data)
        out_file.write(f'epoch={epoch} crossentropy(train): {ce_train:.10f}\n')
        out_file.write(
            f'epoch={epoch} crossentropy(validation): {ce_valid:.10f}\n')
    error_train = write_preds(net, train_data, train_out)
    out_file.write(f'error(train): {error_train}\n')
    error_valid = write_preds(net, valid_data, valid_out)
    out_file.write(f'error(validation): {error_valid}\n')
    return net


if __name__ == '__main__':
    assert len(sys.argv) == 10
    train_input = sys.argv[1]
    valid_input = sys.argv[2]
    train_out = sys.argv[3]
    valid_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hid_units = int(sys.argv[7])
    init_flag = sys.argv[8]
    lr = float(sys.argv[9])

    train_data = load_data(train_input)
    dataset_args = {'num_classes': train_data.num_classes}
    valid_data = load_data(valid_input, dataset_args)
    model = train_model(train_data, valid_data, hid_units,
                        lr, num_epoch, init_flag,
                        metrics_out, train_out, valid_out)