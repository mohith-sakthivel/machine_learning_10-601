import os
import sys
import csv
import math


def open_write_path(file_path, open_mode):
    dir_path = os.path.dirname(file_path)
    if not (os.path.exists(dir_path) or dir_path == ''):
        os.makedirs(os.path.dirname(file_path))
    return open(file_path, open_mode)


def inspect_dataset(input_file, output_file):
    labels = {}
    total_count = 0
    with open(input_file, 'r') as in_file:
        tsv_reader = csv.reader(in_file, delimiter='\t')
        _ = next(tsv_reader)    # remove header
        for line in tsv_reader:
            total_count += 1
            if line[-1] in labels.keys():
                labels[line[-1]] += 1
            else:
                labels[line[-1]] = 1
    out_file = open_write_path(output_file, 'w')
    error = (1 - (max(labels.values()) / total_count))
    entropy = sum([-(count/total_count) * math.log((count/total_count), 2)
                        for count in labels.values()])
    out_file.write('entropy: %0.12f\n'%entropy)
    out_file.write('error: %0.12f'%error)
    out_file.close()



if __name__=='__main__':
    assert len(sys.argv) == 3
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    inspect_dataset(input_file, output_file)