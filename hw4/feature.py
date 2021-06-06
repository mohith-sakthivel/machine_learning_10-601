import os
import sys
import csv

TRIM_THRESHOLD = 4


def get_dict(path):
    """Return the token dict by reading the dict file"""
    token_dict = {}
    with open(path, 'r') as dict_file:
        reader = csv.reader(dict_file, delimiter=' ')
        for line in reader:
            assert len(line) == 2
            token_dict[line[0]] = int(line[1])
    return token_dict


def writer_1(token_counts, out_file):
    """Writes formatted output as required by model-1"""
    for index, count in token_counts.items():
        out_file.write('%d:1\t' % index)


def writer_2(token_counts, out_file):
    """Write formatted output as required by model-2"""
    for index, count in token_counts.items():
        if count < TRIM_THRESHOLD:
            out_file.write('%d:1\t' % index)


def process_data(in_file, token_dict, out_file, type_flag):
    """
    Read data from an input file and process into a 
    suitable format for training/inference
    """
    labels = []
    features = []
    # read data from input file
    with open(in_file, 'r') as in_f:
        reader = csv.reader(in_f, delimiter='\t')
        for line in reader:
            labels.append(line[0])
            feature = {}
            for token in line[1].split(' '):
                if token in token_dict.keys():
                    feat_id = token_dict[token]
                    feature[feat_id] = feature.get(feat_id, 0) + 1
            features.append(feature)
    # write processed outputs
    out_dir = os.path.dirname(out_file)
    if not (out_dir == '' or os.path.exists(out_dir)):
        os.makedirs(out_dir)
    writer_fn = writer_1 if type_flag == 1 else writer_2
    with open(out_file, 'w') as out_f:
        for label, feat in zip(labels, features):
            out_f.write(label + '\t')
            writer_fn(feat, out_f)
            out_f.write('\n')


if __name__ == '__main__':
    # python feature.py handout/smalldata/train_data.tsv handout/smalldata/valid_data.tsv handout/smalldata/test_data.tsv handout/dict.txt try/train.tsv try/valid.tsv try/test.tsv 1
    # python feature.py handout/largedata/train_data.tsv handout/largedata/valid_data.tsv handout/largedata/test_data.tsv handout/dict.txt try/train.tsv try/valid.tsv try/test.tsv 1
    assert len(sys.argv) == 9
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3]
    dict_input = sys.argv[4]
    formatted_train_out = sys.argv[5]
    formatted_validation_out = sys.argv[6]
    formatted_test_out = sys.argv[7]
    feature_flag = int(sys.argv[8])

    if feature_flag == 1:
        pass
    elif feature_flag == 2:
        pass
    else:
        raise ValueError

    token_dict = get_dict(dict_input)
    process_data(train_input, token_dict, formatted_train_out, feature_flag)
    process_data(validation_input, token_dict,
                 formatted_validation_out, feature_flag)
    process_data(test_input, token_dict, formatted_test_out, feature_flag)
