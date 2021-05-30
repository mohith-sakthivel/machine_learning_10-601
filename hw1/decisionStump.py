import os
import sys
import csv


class DecisionStump:

    def __init__(self, split_index):
        self._split_index = split_index
        self._attribute_vals = {}

    def update_tree(self, sample):
        key_value = sample[self._split_index]
        label = sample[-1]
        if key_value in self._attribute_vals.keys():
            if label in self._attribute_vals[key_value].keys():
                self._attribute_vals[key_value][label] += 1
            else:
                self._attribute_vals[key_value][label] = 1
        else:
            self._attribute_vals[key_value] = {label: 1}
    
    def predict(self, sample):
        key_value = sample[self._split_index]
        max_label = None
        max_value = 0
        for label, count  in self._attribute_vals[key_value].items():
            if count > max_value or max_label is None:
                max_label = label
                max_value = count
        return max_label


def get_stump(train_input, split_index):
    decision_stump = DecisionStump(split_index)
    with open(train_input) as train_tsv:
        tsv_reader = csv.reader(train_tsv, delimiter='\t')
        header = next(tsv_reader)
        num_samples = 0
        for line in tsv_reader:
            num_samples += 1
            decision_stump.update_tree(line)
    print('Trained on %d samples' % num_samples)
    return decision_stump

def open_write_path(file_path, open_mode):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    return open(file_path, open_mode)

def generate_metrics(d_stump, train_input, test_input, train_out, test_out, metrics_out):
    set_1 = (train_input, train_out, 'train')
    set_2 = (test_input, test_out, 'test')
    metrics_file = open_write_path(metrics_out, 'w')
    for (in_path, out_path, tag) in [set_1, set_2]:
        out_file = open_write_path(out_path, 'w')
        with open(in_path) as in_file:
            tsv_reader = csv.reader(in_file, delimiter='\t')
            header = next(tsv_reader)
            num_samples = 0
            error = 0
            for line in tsv_reader:
                num_samples += 1
                label = d_stump.predict(line)
                out_file.write(label + '\n')
                if label != line[-1]:
                    error += 1
        out_file.close()
        metrics_file.write('error(%s): %f\n' % (tag, (error/num_samples)))
    metrics_file.close()


if __name__=='__main__':
    assert len(sys.argv) == 7
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    split_index = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

    d_stump = get_stump(train_input, split_index)
    generate_metrics(d_stump, train_input, test_input, train_out, test_out, metrics_out)

