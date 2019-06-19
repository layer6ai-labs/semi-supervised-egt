import numpy as np
import pickle
import pdb
import pandas as pd
import matplotlib.pyplot as plt
from scipy import misc
import os
from tqdm import tqdm
from collections import defaultdict
import argparse
import operator
import faiss
import time


parser = argparse.ArgumentParser()
parser.add_argument('--all-considered', type=bool, default=True)
parser.add_argument('--ransac', type=bool, default=False)
# parser.add_argument('--recurse', type=bool, default=False)
# parser.add_argument('--num-iter', type=int, default=1)
# parser.add_argument('--add', type=bool, default=True)

args = parser.parse_args()


def build_train_to_label_dict(train_csv_path):
    train_to_label = {}
    train_df = pd.read_csv(train_csv_path)
    train_to_label = train_df.set_index('id')['landmark_id'].to_dict()
    return train_to_label


def read_index_to_train(path, train_to_label, proc_type):
    num_items_considered = 3
    index_to_train = {}
    index_to_label = {}
    index_to_label_score = {}
    index_to_train_scores = {}
    max_score = 0
    total = 0
    with open(path, 'r') as f:
        for line in tqdm(f):
            split_line = line.split(',')
            index_hash = split_line[0]
            # train_hash, score = split_line[1].split(' ')[0], split_line[1].split(' ')[1]
            items = split_line[1].split(' ')

            # score = int(score) / 1000000.0
            # if score > 0.90:
            #     index_to_train[index_hash] = train_hash
            #     index_to_label[index_hash] = train_to_label[train_hash]

            # if index_hash == 'ff4135c3071f7b36':
            #     breakpoint()

            # thresh = 700000
            # default_thresh = 800000
            if proc_type == 'index':
                thresh = 700000
                default_thresh = 800000
            elif proc_type == 'test':
                thresh = 1400000
                default_thresh = 1600000
            else:
                raise(Exception)
                exit(-1)

            if not args.all_considered:
                item_hashes = []
                item_scores = []
                for i in range(num_items_considered):
                    item_hashes.append(items[2 * i])
                    item_scores.append(int(items[2 * i + 1]))

                if item_scores[0] > thresh and item_scores[1] > thresh and item_scores[2] > thresh:
                    if train_to_label[item_hashes[0]] == train_to_label[item_hashes[1]] or train_to_label[item_hashes[0]] == train_to_label[item_hashes[2]]:
                        index_to_label[index_hash] = train_to_label[item_hashes[0]]
                        index_to_train[index_hash] = item_hashes
                        index_to_label_score[index_hash] = max([item_scores[0], item_scores[1]]) if train_to_label[item_hashes[0]] == train_to_label[item_hashes[1]] else max([item_scores[0], item_scores[2]])
                    elif train_to_label[item_hashes[1]] == train_to_label[item_hashes[2]]:
                        index_to_label[index_hash] = train_to_label[item_hashes[1]]
                        index_to_train[index_hash] = item_hashes
                        index_to_label_score[index_hash] = max([item_scores[1], item_scores[2]])
                    else: # todo: hacky
                        index_to_train[index_hash] = item_hashes
                        index_to_label_score[index_hash] = default_thresh

            else:
                if proc_type == 'index':
                    voting_candidates = 5
                    majority_candidates = 3
                elif proc_type == 'test':
                    voting_candidates = 3
                    majority_candidates = 2
                else:
                    raise (Exception)
                    exit(-1)

                item_hashes = []
                item_scores = []
                for i in range(voting_candidates):
                    item_hashes.append(items[2 * i])
                    item_scores.append(int(items[2 * i + 1]))

                all_class_dict = defaultdict(int)
                all_class_highest_score = defaultdict(int)
                for idx, h in enumerate(item_hashes):
                    class_h = train_to_label[h]
                    all_class_dict[class_h] += 1
                    all_class_highest_score[class_h] = max(all_class_highest_score[class_h], item_scores[idx])
                match = max(all_class_dict.items(), key=operator.itemgetter(1))
                if match[1] >= majority_candidates:
                    index_to_label[index_hash] = match[0]
                    index_to_label_score[index_hash] = all_class_highest_score[index_to_label[index_hash]]
                else: # todo: hacky
                    index_to_label_score[index_hash] = default_thresh
                index_to_train[index_hash] = item_hashes

    return index_to_train, index_to_label, index_to_label_score, index_to_train_scores


def read_index_to_train_multi_label(path, train_to_label):
    num_items_considered = 5
    index_to_train = defaultdict(list)
    index_to_label = defaultdict(set)
    index_to_label_score = {}
    max_score = 0
    total = 0
    with open(path, 'r') as f:
        for line in f:
            split_line = line.split(',')
            index_hash = split_line[0]
            # train_hash, score = split_line[1].split(' ')[0], split_line[1].split(' ')[1]
            items = split_line[1].split(' ')
            item_hashes = []
            item_scores = []
            for i in range(num_items_considered):
                item_hashes.append(items[2 * i])
                item_scores.append(int(items[2 * i + 1]))

            # score = int(score) / 1000000.0
            # if score > 0.90:
            #     index_to_train[index_hash] = train_hash
            #     index_to_label[index_hash] = train_to_label[train_hash]

            if item_scores[0] > 700000 and item_scores[1] > 700000 and item_scores[2] > 700000:
                for i in range(num_items_considered):
                    index_to_label[index_hash].add(train_to_label[item_hashes[i]])
                    index_to_train[index_hash].extend(item_hashes)
                    index_to_label_score[index_hash] = max(item_scores[:num_items_considered])

    return index_to_train, index_to_label, index_to_label_score


def dump_pickle(fname, data):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def read_submission_file(s_path):

    with open(s_path, 'r') as f:
        data = f.readlines()

    data = data[1:]
    data = [elem.strip() for elem in data]
    if data[-1] == '':
        data = data[:-1]

    return data


def parse_submission_data(s_data, test_to_label):
    submission_dict = {}
    for line in s_data:
        test_hash = line.split(',')[0]
        if test_hash in test_to_label:
            submission_dict[test_hash] = line.split(',')[1].split(' ')

    return submission_dict


def get_full_submission_dict(s_data):
    s_dict = {}
    for line in s_data:
        test_hash = line.split(',')[0]
        s_dict[test_hash] = line.split(',')[1].split(' ')
    return s_dict


def build_label_to_index_dict(index_to_label):
    label_to_index = {}
    for index_hash in index_to_label:
        label = index_to_label[index_hash]
        if type(label) == list:
            for l in label:
                if l in label_to_index:
                    label_to_index[l].append(index_hash)
                else:
                    label_to_index[l] = [index_hash]
        else:
            if label in label_to_index:
                label_to_index[label].append(index_hash)
            else:
                label_to_index[label] = [index_hash]

    return label_to_index


def build_same_label_list2(submission_dict, test_to_label, index_to_label, label_to_index):

    same_label_sub_dict = {}
    for key in submission_dict:
        if key not in test_to_label:
            same_label_sub_dict[key] = []
            continue
        label = test_to_label[key]
        sub_list = submission_dict[key]
        same_label_sub_dict[key] = []
        for index_hash in sub_list:
            if index_hash in index_to_label and index_to_label[index_hash] == label:
                same_label_sub_dict[key].append(index_hash)

    return same_label_sub_dict


def build_same_label_list(submission_dict, test_to_label, index_to_label, label_to_index):
    same_label_sub_dict = {}
    other_index_same_label_dict = {}
    for key in submission_dict:
        label = test_to_label[key]
        sub_list = submission_dict[key]
        same_label_sub_dict[key] = []
        for index_hash in sub_list:
            if index_hash in index_to_label and index_to_label[index_hash] == label:
                same_label_sub_dict[key].append(index_hash)

    for key in submission_dict:
        label = test_to_label[key]
        if label in label_to_index:
            # same_label_list = set(label_to_index[label])
            # sub_list = set(submission_dict[key])
            same_label_list = label_to_index[label]
            sub_list = submission_dict[key]
            diff = []
            for item in same_label_list:
                if item not in sub_list:
                    diff.append(item)
            # diff = same_label_list - sub_list
            other_index_same_label_dict[key] = list(diff)

    return same_label_sub_dict, other_index_same_label_dict




def visualize(same_label_sub_dict, other_index_same_label_dict):

    test_base_path = '/data/landmark/stage2/test/'

    for q, test_hash in enumerate(same_label_sub_dict):
        rows = 5
        columns = 10
        num_images_to_show = rows * columns
        plotted = 1
        fig = plt.figure(q, figsize(16,10))
        fig.subplots_adjust(left=0.05, right=0.95, bottom = 0.05, top=0.95, wspace=0.4, hspace=0.4)
        top, bottom, left, right = [10]*4
        image_size = 300

        img = misc.imread('')



def read_pickle_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data



def write_new_submission_file(s_dict, fname):

    with open(fname, 'w') as f:
        f.write('id,images\n')
        for key in s_dict:
            line = key + ','
            arr = s_dict[key]
            line += ' '.join(arr)
            line += '\n'
            f.write(line)


def write_jason_prebuild(sdict, fname):

    with open(fname, 'w') as f:
        for key in sdict:
            if len(sdict[key]) > 0:
                line = key + ','
                arr = sdict[key]
                mystr = ''
                for item in arr:
                    mystr += item + ' ' + str(1000000) + ' '
                line += mystr + '\n'
                f.write(line)


def write_new_submission_file(s_dict, fname):

    with open(fname, 'w') as f:
        f.write('id,images\n')
        for key in s_dict:
            line = key + ','
            arr = s_dict[key]
            line += ' '.join(arr)
            line += '\n'
            f.write(line)


def write_to_submission_without_rerank(test_to_label, label_to_index):
    sdict = {}
    for t, label in tqdm(test_to_label.items()):
        if label in label_to_index:
            sdict[t] = label_to_index[label]

    return sdict


def union_with_ransac(a_to_b, ransac_a_to_b):

    set_diff = set(ransac_a_to_b.keys()) - set(a_to_b.keys())

    for key in set_diff:
        a_to_b[key] = ransac_a_to_b[key]

    return a_to_b


def sort_indices(label_to_index, index_to_label_score):
    sorted_label_to_index = defaultdict(list)
    for label in label_to_index:
        for h in label_to_index[label]:
            sorted_label_to_index[label].append((h, index_to_label_score[h]))
        sorted_label_to_index[label] = sorted(sorted_label_to_index[label], key=lambda k: k[1], reverse=True)
        sorted_label_to_index[label] = [x[0] for x in sorted_label_to_index[label]]
    return sorted_label_to_index


def main():

    print("train to label")
    train_to_label = build_train_to_label_dict('data/train.csv')
    dump_pickle('data/train_to_label.pkl', train_to_label)

    print("index to label")
    index_to_train, index_to_label, index_to_label_score, index_to_train_scores = read_index_to_train('data/index_to_train_concat.txt', train_to_label, 'index')

    dump_pickle('data/index_to_label.pkl', index_to_label)
    dump_pickle('data/index_to_train.pkl', index_to_train)
    dump_pickle('data/index_to_label_score.pkl', index_to_label_score)

    print("test to label")
    test_to_train, test_to_label, test_to_label_score, test_to_train_scores = read_index_to_train('data/test_to_train_concat.txt', train_to_label, 'test')

    dump_pickle('data/test_to_label.pkl', test_to_label)
    dump_pickle('data/test_to_train.pkl', test_to_train)
    dump_pickle('data/test_to_label_score.pkl', test_to_label_score)

    if args.ransac:
        test_to_label_ransac = read_pickle_file('data/test_to_label_ransac.pkl')
        index_to_label_ransac = read_pickle_file('data/index_to_label_ransac.pkl')

        test_to_label_old = test_to_label
        index_to_label_old = index_to_label

        test_to_label = union_with_ransac(test_to_label, test_to_label_ransac)
        index_to_label = union_with_ransac(index_to_label, index_to_label_ransac)

        test_inter = set(test_to_label_ransac).intersection(test_to_label_old)
        index_inter = set(index_to_label_ransac).intersection(index_to_label_old)

        for t in list(test_inter):
            if test_to_label_old[t] != test_to_label_ransac[t]:
                test_to_label[t] = [test_to_label_old[t], test_to_label_ransac[t]]

        for i in list(index_inter):
            if index_to_label_old[i] != index_to_label_ransac[i]:
                index_to_label[i] = [index_to_label_old[i], index_to_label_ransac[i]]

    print("label to index")
    label_to_index = build_label_to_index_dict(index_to_label)
    dump_pickle('data/label_to_index.pkl', label_to_index)

    label_to_test = build_label_to_index_dict(test_to_label)
    dump_pickle('data/label_to_test.pkl', label_to_test)

    label_to_index = sort_indices(label_to_index, index_to_label_score)
    pickle.dump(label_to_index, open("data/label_to_index_sorted.pkl", "wb"))

    print('num in test to label', len(test_to_label))
    print('num in index to label', len(index_to_label))


if __name__ == "__main__":
    main()

