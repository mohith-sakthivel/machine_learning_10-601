import os
import csv
import sys
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 0.1


def get_dict(path):
    """Return the token dict by reading the dict file"""
    feature_dict = {}
    with open(path, 'r') as dict_file:
        reader = csv.reader(dict_file, delimiter=' ')
        for line in reader:
            assert len(line) == 2
            feature_dict[line[0]] = int(line[1])
    return feature_dict


class DataSet:
    """
    Object for storing label and feature data
    """

    def __init__(self):
        self._labels = []
        self._features = []

    def __len__(self):
        return len(self._labels)

    def append(self, label, feat):
        self._labels.append(label)
        self._features.append(feat)

    def __iter__(self):
        return zip(self._labels, self._features)


def load_data(path):
    """Load formatted input data from a file"""
    with open(path, 'r') as read_file:
        reader = csv.reader(read_file, delimiter='\t')
        data = DataSet()
        for line in reader:
            label = int(line[0])
            feature = []
            if len(line) > 1:
                for feat_tag in line[1:]:
                    feature.append(int(feat_tag[:-2]))
            data.append(label, feature)
    return data


def open_write_path(file_path, open_mode):
    dir_path = os.path.dirname(file_path)
    if not (os.path.exists(dir_path) or dir_path == ''):
        os.makedirs(os.path.dirname(file_path))
    return open(file_path, open_mode)


def get_grad(theta, label, feature):
    theta_x = np.dot(theta, feature)
    theta_grad = -(label - (np.exp(theta_x)/(1 + np.exp(theta_x)))) * feature
    return theta_grad


def process_feature(raw_feat, num_tokens):
    """Create dense feature from sparse feature"""
    processed_feat = np.zeros(num_tokens + 1, dtype=np.float32)
    processed_feat[raw_feat + [num_tokens, ]] = 1
    return processed_feat

def calculate_nll(theta, dataset):
    nll = []
    for label, feat in dataset:
        theta_x = np.dot(theta, process_feature(feat, len(theta) - 1))
        nll.append(-label * theta_x + np.log(1 + np.exp(theta_x)))
    return np.mean(nll)


def train_model(train_data, valid_data, token_dict, num_epochs, track_nll=False):
    num_tokens = len(token_dict)
    theta = np.zeros(num_tokens+1, dtype=np.float32)
    if track_nll:
        train_nll = []
        valid_nll = []
    for _ in range(1, num_epochs+1):
        for label, feature in train_data:
            theta_grad = get_grad(theta, label,
                                  process_feature(feature, num_tokens))
            theta -= LEARNING_RATE * theta_grad / len(train_data)
        if track_nll:
            train_nll.append(calculate_nll(theta, train_data))
            valid_nll.append(calculate_nll(theta, valid_data))
    if track_nll:
        plt.plot(np.arange(1, num_epochs+1), train_nll, 'r', label='train')
        plt.plot(np.arange(1, num_epochs+1), valid_nll, 'b', label='validation')
        plt.xlabel('epochs')
        plt.ylabel('negative log-likelihood')
        plt.legend()
        plt.savefig('stat_curve.png')

    return theta


def eval_model(theta, dataset, out_path):
    """Evaluate model predictions on the dataset"""
    out_file = open_write_path(out_path, 'w')
    error = 0
    for label, feature in dataset:
        processed_feature = process_feature(feature, len(theta)-1)
        pred = 1/(1 + np.exp(-np.dot(theta, processed_feature)))
        if np.round(pred) != label:
            error += 1
        out_file.write('%d\n'%(np.round(pred)))
    out_file.close()
    return error/len(dataset)


def generate_metrics(theta, train_data, test_data, train_out, test_out, metrics_out):
    datasets = [train_data, test_data]
    out_paths = [train_out, test_out]
    tags = ['train', 'test']

    metrics_file = open_write_path(metrics_out, 'w')
    for data, out_path, tag in zip(datasets, out_paths, tags):
        error_rate = eval_model(theta, data, out_path)
        metrics_file.write('error(%s): %0.6f\n' % (tag, error_rate))
    metrics_file.close()


if __name__ == '__main__':
    # python lr.py handout/smalloutput/model1_formatted_train.tsv handout/smalloutput/model1_formatted_valid.tsv handout/smalloutput/model1_formatted_test.tsv handout/dict.txt try/train.tsv try/valid.tsv try/test.tsv 60
    assert len(sys.argv) == 9
    formatted_train_input = sys.argv[1]
    formatted_validation_input = sys.argv[2]
    formatted_test_input = sys.argv[3]
    dict_input = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epochs = int(sys.argv[8])

    token_dict = get_dict(dict_input)
    train_data = load_data(formatted_train_input)
    valid_data = load_data(formatted_validation_input)
    test_data = load_data(formatted_test_input)

    theta = train_model(train_data, valid_data,
                        token_dict, num_epochs, track_nll=False)
    generate_metrics(theta, train_data, test_data,
                     train_out, test_out, metrics_out)
