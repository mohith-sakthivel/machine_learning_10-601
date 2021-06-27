import os
import sys
import csv

import numpy as np


class DataSet:
    """
    Object for storing label and feature data
    """

    def __init__(self, feature_dim=None):
        self._class = []
        self._features = []
        self.label_id = {'tool': 0, 'building': 1}
        self.id_label = {0: 'tool', 1: 'building'}
        self._feature_dim = feature_dim

    def __len__(self):
        return len(self._class)

    def append(self, label, feat):
        self._class.append(self.label_id[label])
        self._features.append(feat)

    @property
    def feature_dim(self):
        if self._feature_dim is None:
            self._feature_dim = len(self._features[0])
        return self._feature_dim

    @property
    def num_classes(self):
        return len(self.label_id)

    def __iter__(self):
        return zip(self._class, self._features)


def load_data(path, dataset_kwargs={}):
    """Load formatted input data from a file"""
    with open(path, 'r') as read_file:
        reader = csv.reader(read_file)
        data = DataSet(**dataset_kwargs)
        _ = next(reader)
        for line in reader:
            label = str(line[-1])
            feature = [float(i) for i in line[:-1]]
            data.append(label, feature)
    return data


def open_write_path(file_path, open_mode):
    dir_path = os.path.dirname(file_path)
    if not (os.path.exists(dir_path) or dir_path == ''):
        os.makedirs(os.path.dirname(file_path))
    return open(file_path, open_mode)


def get_top_features(cpd_mean, num_features):
    separation = np.abs(cpd_mean[0] - cpd_mean[1])
    return np.argsort(separation)[-num_features:]


def train_model(data, num_voxels):
    """Train a Gaussian Naive Bayes Classifier"""
    prior = np.zeros(data.num_classes,dtype=np.float32)
    cpd_mean = np.zeros((data.num_classes, data.feature_dim), dtype=np.float32)
    cpd_var = np.zeros((data.num_classes, data.feature_dim), dtype=np.float32)

    # calculate model parameters
    for class_id, feat in data:
        prior[class_id] += 1
        cpd_mean[class_id] += feat
    cpd_mean = cpd_mean/prior.reshape((-1, 1))

    for class_id, feat in data:
        cpd_var[class_id] += np.power((feat - cpd_mean[class_id]), 2)
    cpd_var = cpd_var/prior.reshape((-1, 1))

    prior /= len(data)

    return {'prior': prior,
            'cpd_mean': cpd_mean,
            'cpd_var': cpd_var,
            'top_feat': get_top_features(cpd_mean, num_voxels)}


def predict(model, feat):
    """ Give model prediction for one sample """
    feat = np.array(feat, dtype=np.float32)[model['top_feat']]
    cpd_mean = model['cpd_mean'][..., model['top_feat']]
    cpd_var = model['cpd_var'][..., model['top_feat']]

    log_probs = (-0.5 * np.log(2 * np.pi * cpd_var) 
                      - (np.power(feat - cpd_mean, 2) / (2 * cpd_var)))
        
    probs = np.log(model['prior']) + log_probs.sum(axis=-1)
    return np.argmax(probs)


def generate_metrics(model, train_data, test_data, train_out, test_out, metrics_out):
    """Calcualte the error and predicitons over the test and train datasets"""
    evals = [('train', train_data, train_out), ('test', test_data, test_out)]
    metrics = {}

    # get model predictions on test and train data
    for tag, data, out_file in evals:
        out_f = open_write_path(out_file, 'w')
        error = 0
        for class_id, feat in data:
            pred = predict(model, feat)
            out_f.write(data.id_label[pred] + '\n')
            if pred != class_id:
                error += 1
        out_f.close
        metrics[tag] = error/len(data)

    # write error metrics on test and train data
    out_file = open_write_path(metrics_out, 'w')
    for tag, val in metrics.items():
        out_file.write('error(%s): %0.6f\n' % (tag, val))
    out_file.close()


if __name__ == "__main__":
    assert len(sys.argv) == 7
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_voxels = int(sys.argv[6])

    # train_input = 'handout/data/train_data.csv'
    # test_input = 'handout/data/test_data.csv'
    # train_out = 'try/train.labels'
    # test_out = 'try/test.labels'
    # metrics_out = 'try/metrics.txt'
    # num_voxels = 21764

    train_data = load_data(train_input)
    test_data = load_data(test_input)

    model = train_model(train_data, num_voxels)
    generate_metrics(model, train_data, test_data,
                     train_out, test_out, metrics_out)
    print('done')
