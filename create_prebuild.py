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

import sys
sys.path.append('..')

from teststuff import runevaluation


parser = argparse.ArgumentParser()
parser.add_argument('--load', type=bool, default=False)
parser.add_argument('--multi', type=bool, default=False)
parser.add_argument('--all-considered', type=bool, default=True)
parser.add_argument('--ransac', type=bool, default=False)
parser.add_argument('--recurse', type=bool, default=False)
parser.add_argument('--num-iter', type=int, default=1)
parser.add_argument('--add', type=bool, default=True)

args = parser.parse_args()


def doRetrieval(Q, X, k=100, verbose=True):
    res = faiss.StandardGpuResources()
    if verbose:
        print("creating indexFlatl2")
    index = faiss.IndexFlatL2(X.shape[1])
    if verbose:
        print("put to gpu")
    index = faiss.index_cpu_to_gpu(res, 0, index)
    if verbose:
        print("adding index to faiss")

    # X shape: nxd
    # split into 2 chunks
    index.add(X)
    if verbose:
        print("num of index: " + str(index.ntotal))
    if verbose:
        print("searching")
    start = time.time()
    D, I = index.search(Q, k)
    if verbose:
        print('Computing dot product')
    elapse = time.time() - start
    if verbose:
        print(elapse)

    return D, I


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

                    if args.recurse:
                        index_to_train_scores[index_hash] = item_scores
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

                if args.recurse:
                    index_to_train_scores[index_hash] = item_scores

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


def build_label_to_index_dict_multi(index_to_label):
    label_to_index = defaultdict(list)
    for index_hash in index_to_label:
        label_set = index_to_label[index_hash]
        for label in label_set:
            label_to_index[label].append(index_hash)

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


def get_full_submission_dict(s_data):
    s_dict = {}
    for line in s_data:
        test_hash = line.split(',')[0]
        s_dict[test_hash] = line.split(',')[1].split(' ')
    return s_dict


def sort_indices(label_to_index, index_to_label_score):
    sorted_label_to_index = defaultdict(list)
    for label in label_to_index:
        for h in label_to_index[label]:
            sorted_label_to_index[label].append((h, index_to_label_score[h]))
        sorted_label_to_index[label] = sorted(sorted_label_to_index[label], key=lambda k: k[1], reverse=True)
        sorted_label_to_index[label] = [x[0] for x in sorted_label_to_index[label]]
    return sorted_label_to_index


def prepend_all(test_to_label, label_to_index, index_to_label_score, index_to_label):
    lines = list(open('data/jason_old_best.txt', 'r'))
    lines = [l.strip() for l in lines[1:]]
    s_dict = {}
    for line in lines:
        test_hash = line.split(',')[0]
        s_dict[test_hash] = line.split(',')[1].split(' ')

    # sort
    label_to_index = sort_indices(label_to_index, index_to_label_score)
    # pickle.dump(label_to_index, open("data/label_to_index_final.pkl", "wb"))

    # prepend
    prepend_dict = defaultdict(list)
    if not args.multi:
        for q, label in test_to_label.items():
            if type(label) == list:
                for l in label:
                    if l in label_to_index:
                        prepend_dict[q].extend(label_to_index[l])
            else:
                if label in label_to_index:
                    prepend_dict[q].extend(label_to_index[label])
    else:
        for q, q_label_set in tqdm(test_to_label.items()):
            for i, i_label_set in index_to_label.items():
                inter = q_label_set.intersection(i_label_set)
                if len(inter) > 1:
                    for l in inter:
                        prepend_dict[q].extend(label_to_index[l])

    print("length of queries changed is ", len(prepend_dict))

    # aggregate
    new_sub_dict = defaultdict(list)
    for q in tqdm(s_dict):
        if q in prepend_dict:
            new_sub_dict[q].extend(prepend_dict[q])
        for s_i in s_dict[q]:
            if s_i not in new_sub_dict[q]:
                new_sub_dict[q].append(s_i)

    # cut
    for key in new_sub_dict:
        new_sub_dict[key] = new_sub_dict[key][:100]

    for key in new_sub_dict:
        assert (len(new_sub_dict[key]) == 100)

    # write to submission file
    write_new_submission_file(new_sub_dict, 'data/rerank_old_jason_new_heursitic_and_prepend_lower_thres_new_voting_union_ransac.txt')
    runevaluation()


def main():

    if not args.load:
        print("train to label")
        train_to_label = build_train_to_label_dict('/data/landmark/train.csv')
        dump_pickle('data/train_to_label.pkl', train_to_label)

        print("index to label")
        if not args.multi:
            index_to_train, index_to_label, index_to_label_score, index_to_train_scores = read_index_to_train('index_to_train_concat.txt', train_to_label, 'index')
        else:
            index_to_train, index_to_label, index_to_label_score = read_index_to_train_multi_label('index_to_train.txt',
                                                                                       train_to_label)
        dump_pickle('data/index_to_label.pkl', index_to_label)
        dump_pickle('data/index_to_train.pkl', index_to_train)
        dump_pickle('data/index_to_label_score.pkl', index_to_label_score)

        print("test to label")
        if not args.multi:
            test_to_train, test_to_label, test_to_label_score, test_to_train_scores = read_index_to_train('test_to_train_concat.txt', train_to_label, 'test')
        else:
            test_to_train, test_to_label, test_to_label_score = read_index_to_train_multi_label('test_to_train.txt', train_to_label)

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
        if not args.multi:
            label_to_index = build_label_to_index_dict(index_to_label)
        else:
            label_to_index = build_label_to_index_dict_multi(index_to_label)
        dump_pickle('data/label_to_index.pkl', label_to_index)

        if not args.multi:
            label_to_test = build_label_to_index_dict(test_to_label)
        else:
            label_to_test = build_label_to_index_dict_multi(test_to_label)
        dump_pickle('data/label_to_test.pkl', label_to_test)

    else:
        test_to_label = read_pickle_file('data/test_to_label.pkl')
        index_to_label = read_pickle_file('data/index_to_label.pkl')

        # test_to_label = read_pickle_file('data/test_to_label.pkl')
        # index_to_label = read_pickle_file('data/index_to_label.pkl')

        label_to_test = read_pickle_file('data/label_to_test.pkl')
        label_to_index = read_pickle_file('data/label_to_index.pkl')

        # label_to_test = build_label_to_index_dict(test_to_label)
        # label_to_index = build_label_to_index_dict(index_to_label)

        test_to_label_score = read_pickle_file('data/test_to_label_score.pkl')
        index_to_label_score = read_pickle_file('data/index_to_label_score.pkl')

    print('num in test to label', len(test_to_label))
    print('num in index to label', len(index_to_label))


    ###########################################
    # visualize index
    # for t in index_to_train.keys():
    #     print('---------------------------------------------------------------------')
    #     os.system('display /data/landmark/stage2/index/' + t[0] + '/' + t[1] + '/' + t[2] + '/' + t + '.jpg')
    #     print("starting index vis")
    #     for e in index_to_train[t][:3]:
    #         os.system('display /data/landmark/train/' + e[0] + '/' + e[1] + '/' + e[2] + '/' + e + '.jpg')

    # visualize test
    # for t in test_to_train.keys():
    #     print('---------------------------------------------------------------------')
    #     os.system('display /data/landmark/stage2/test/' + t[0] + '/' + t[1] + '/' + t[2] + '/' + t + '.jpg')
    #     print("starting index vis")
    #     for e in test_to_train[t][:3]:
    #         os.system('display /data/landmark/train/' + e[0] + '/' + e[1] + '/' + e[2] + '/' + e + '.jpg')

    # train_to_label = read_pickle_file('data/train_to_label.pkl')
    # index_to_label = read_pickle_file('data/index_to_label.pkl')
    # test_to_label = read_pickle_file('data/test_to_label.pkl')
    # label_to_index = read_pickle_file('data/label_to_index.pkl')
    # label_to_test = read_pickle_file('data/label_to_test.pkl')
    # test_to_train = read_pickle_file('data/test_to_train.pkl')
    # index_to_train = read_pickle_file('data/index_to_train.pkl')

    prepend_all(test_to_label, label_to_index, index_to_label_score, index_to_label)

    if args.recurse:
        print("loading features")
        Q_features = np.load("/data/landmark/stage2/ss_gem_8928_retrieval_qvec_hashord.npy")
        I_features = np.load("/data/landmark/stage2/ss_gem_8928_retrieval_vec_hashord.npy")

        test_hashes = list(open('/data/landmark/stage2/test.csv', 'r'))[1:]
        test_hashes = [l.strip() for l in test_hashes]
        index_hashes = list(open('/data/landmark/stage2/index.csv', 'r'))[1:]
        index_hashes = [l.strip() for l in index_hashes]


        Q_features = Q_features.astype("float32")
        I_features = I_features.astype("float32")
        print("converting to contiguous")
        Q = np.ascontiguousarray(Q_features.T)
        X = np.ascontiguousarray(I_features.T)

        # concat everything for ease
        all_hashes = test_hashes + index_hashes
        test_hashes = set(test_hashes)
        index_hashes = set(index_hashes)
        all_features = np.concatenate([Q, X])
        print(all_features.shape)

        # build hash to location
        hash_to_location = {}
        for i in range(len(all_hashes)):
            hash_to_location[all_hashes[i]] = i
        num_to_keep = 5
        num_votes_required = 4
        for i in range(args.num_iter):
            # do for test first
            q_hashes = []
            qs = []
            for test in test_hashes:
                if test not in test_to_label:
                    q = all_features[hash_to_location[test], :]
                    qs.append(q)
                    q_hashes.append(test)

            # do for index too:
            # track their hashes
            for index in index_hashes:
                if index not in index_to_label:
                    q = all_features[hash_to_location[index], :]
                    qs.append(q)
                    q_hashes.append(index)

            qs = np.array(qs)
            get_index = [key for key in index_to_label]
            get_test = [key for key in test_to_label]
            get_all = get_test + get_index
            xs = [hash_to_location[name] for name in get_all]
            x = all_features[xs, :]

            # do retrieval
            _, I = doRetrieval(qs, x, k=num_to_keep, verbose=False)
            print(I.shape)

            # for each one calculate the similarity
            added_q = 0
            added_i = 0
            for i in tqdm(range(I.shape[0])):
                name = q_hashes[i]
                all_scores_for_q = []
                if name in test_hashes:
                    for j in range(len(test_to_train[name])):
                        all_scores_for_q.append((test_to_train_scores[name][j], test_to_train[name][j]))
                elif name in index_hashes:
                    for j in range(len(index_to_train[name])):
                        all_scores_for_q.append((index_to_train_scores[name][j], index_to_train[name][j]))
                else:
                    print("Something is wrong!!!!!!!!!!!")
                for j in range(I.shape[1]):
                    score = int(np.matmul(qs[i], x[I[i, j], :]) * 1000000)
                    if name in test_hashes:
                        all_scores_for_q.append((score, get_all[I[i, j]]))
                    else:
                        all_scores_for_q.append((score, get_all[I[i, j]]))
                # compare with top3 and re-do vote
                if name in test_hashes:
                    all_scores_for_q.sort()
                    all_scores_for_q = all_scores_for_q[-num_to_keep:]
                    votes = {}
                    test_to_train[name] = []
                    max_score = {}
                    for score, c in all_scores_for_q:
                        # update test_to_train and test_to_train_scores
                        test_to_train[name].append(c)

                        if c in train_to_label:
                            lb = train_to_label[c]
                        elif c in index_to_label:
                            lb = index_to_label[c]
                        elif c in test_to_label:
                            lb = test_to_label[c]
                        else:
                            print("somethign wrong here!!!")
                            print(name)
                        if score > 900000:
                            if lb in votes:
                                votes[lb] += 1
                            else:
                                votes[lb] = 1
                            if lb in max_score:
                                max_score[lb] = max(max_score[lb], score)
                            else:
                                max_score[lb] = score
                    lb = -1
                    for key in votes:
                        if votes[key] >= num_votes_required:
                            test_to_label_score[name] = max_score[key]- 800000
                            # print(name, test_top3[name])
                            test_to_label[name] = key
                            if key in label_to_test:
                                label_to_test[key].append(name)
                            else:
                                label_to_test[key] = [name]
                            added_q += 1
                elif name in index_hashes:
                    all_scores_for_q.sort()
                    all_scores_for_q = all_scores_for_q[-num_to_keep:]
                    votes = {}
                    index_to_train[name] = []
                    max_score = {}
                    for score, c in all_scores_for_q:
                        # update test_to_train and test_to_train_scores
                        index_to_train[name].append(c)

                        if c in train_to_label:
                            lb = train_to_label[c]
                        elif c in index_to_label:
                            lb = index_to_label[c]
                        elif c in test_to_label:
                            lb = test_to_label[c]
                        else:
                            print("somethign wrong here!!!")
                            print(name)
                        if score > 900000:
                            if lb in votes:
                                votes[lb] += 1
                            else:
                                votes[lb] = 1
                            if lb in max_score:
                                max_score[lb] = max(max_score[lb], score)
                            else:
                                max_score[lb] = score
                    lb = -1
                    for key in votes:
                        if votes[key] >= num_votes_required:
                            index_to_label_score[name] = max_score[key] - 800000
                            # print(name, test_top3[name])
                            index_to_label[name] = key
                            if key in label_to_index:
                                label_to_index[key].append(name)
                            else:
                                label_to_index[key] = [name]
                            added_i += 1
            prepend_all(test_to_label, label_to_index, index_to_label_score, index_to_label)

    exit(0)

    s_data = read_submission_file('data/jason_old_best.txt')
    submission_dict = parse_submission_data(s_data, test_to_label)
    s_dict = get_full_submission_dict(s_data)

    same_label_sub_dict, other_index_same_label_dict = build_same_label_list(submission_dict, test_to_label,
                                                                             index_to_label, label_to_index)

    # s_data = read_submission_file('data/submission_file.txt')
    # s_data = read_submission_file('data/jason_old_best.txt')
    #
    same_label_sub_dict = build_same_label_list2(s_dict, test_to_label, index_to_label, label_to_index)

    # write_jason_prebuild(same_label_sub_dict, 'data/jason_prebuild.txt')

    # new_submission_dict = {}
    # for key in same_label_sub_dict:
    #     new_submission_dict[key] = same_label_sub_dict[key]
    #     for item in s_dict[key]:
    #         if item not in new_submission_dict[key]:
    #             new_submission_dict[key].append(item)

    new_submission_dict = {}
    for key in same_label_sub_dict:
        new_submission_dict[key] = same_label_sub_dict[key]
        if key in other_index_same_label_dict:
            new_submission_dict[key] = new_submission_dict[key] + other_index_same_label_dict[key]

        for item in s_dict[key]:
            if item not in new_submission_dict[key]:
                new_submission_dict[key].append(item)

    for key in new_submission_dict:
        new_submission_dict[key] = new_submission_dict[key][:100]

    for key in new_submission_dict:
        assert (len(new_submission_dict[key]) == 100)

    write_new_submission_file(new_submission_dict, 'data/rerank_old_jason_new_heursitic_and_prepend_lower_thres_new_voting_union_ransac.txt')


if __name__ == "__main__":
    main()

