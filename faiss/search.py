import numpy as np
import faiss
import os
import pandas as pd
import pickle
from tqdm import tqdm
import time
from chunks_fb_search import doRetrieval_chunk


def doRetrieval_chunk_num(Q, X, num_chunks=5, k=100, verbose=True):
    # X shape: n * d
    # split into 2 chunks
    X_length = X.shape[0]
    chunk_length = X_length//num_chunks + 1
    All_D = []
    All_I = []
    for i in tqdm(range(num_chunks)):
        data = X[i * chunk_length: (i+1) * chunk_length]
        Di, Ii = doRetrieval(Q, data, k=k, verbose=verbose)
        Ii += i * chunk_length
        All_I.append(Ii)
        All_D.append(Di)

    D = np.concatenate(All_D, axis=1)
    I = np.concatenate(All_I, axis=1)
    #r = I.tolist()
    #flat_I = [item for sublist in r for item in sublist]
    #flat_I = list(set(flat_I))
    #chunk_X = X[flat_I, :]
    #newD, newI = doRetrieval(Q, chunk_X, k=k, verbose=verbose)
    #all_I = []
    #for i in range(newI.shape[0]):
    #    r = np.array([flat_I[newI[i,j]] for j in range(k)])
    #    all_I.append(r)
    #all_I = np.array(all_I)
    #print(len(flat_list))
    #exit(0)
    all_I = []


    #all_I = [I[i, doRetrieval(Q[i:i+1,:], X[I[i,:]], k=k, verbose=verbose)[1]] for i in tqdm(range(I.shape[0]))]
    #chunk_Is = Parallel(n_jobs=-1)(delayed(matmul_sorting)(Q[i:i+1,:], X[I[i,:]], k) for i in tqdm(range(I.shape[0])))
    chunk_Is = [matmul_sorting(Q[i:i+1,:], X[I[i,:]], k) for i in tqdm(range(I.shape[0]))]
    for i in range(len(chunk_Is)):
        new_I = I[i, chunk_Is[i][0,:]]
        all_I.append(new_I)
    #for i in tqdm(range(I.shape[0])):
    #    # do search again on these indices
    #    chunk_X = X[I[i,:]]
    #    score = np.matmul(Q[i:i+1,:], chunk_X.T)
    #    chunk_I = np.argsort(-score, axis=1)
    #    chunk_I = chunk_I[:k]
    #    #chunk_D, chunk_I = doRetrieval(Q[i:i+1,:], chunk_X, k=k, verbose=verbose)
    #    new_I = I[i, chunk_I[0,:]]
    #    all_I.append(new_I)
    I = np.array(all_I)
    return D, I


def compute_map(df_index, df_query, ranks):
    df = pd.concat([df_index, df_query])
    hash_landmark_dict = pd.Series(df['landmark_id'].values, index=df['id']).to_dict()
    map = 0
    for q in tqdm(range(len(ranks))):
        correct = 0
        map_at_q = 0
        target_hash = ranks[q][0]
        for k in range(1, 1 + min(len(ranks[q]), 100)):
            pred_hash = ranks[q][k]
            if hash_landmark_dict[pred_hash] == hash_landmark_dict[target_hash]:
                correct += 1
                pk_relk = correct / k
                map_at_q += pk_relk
        df_grouped = df_index.groupby('landmark_id').count()['id']
        target_landmark_id = hash_landmark_dict[target_hash]
        if target_landmark_id in df_grouped:
            m_q = df_grouped[target_landmark_id]
            map_at_q /= min(m_q, 100)
            map += map_at_q
    map /= len(ranks)
    print('validation mAP is {}'.format(map))
    return map


def compute_recall(df_index, df_query, ranks):
    df = pd.concat([df_index, df_query])
    hash_landmark_dict = pd.Series(df['landmark_id'].values, index=df['id']).to_dict()
    recall = 0

    for q in tqdm(range(len(ranks))):
        correct = 0
        target_hash = ranks[q][0]
        for k in range(1, 1 + min(len(ranks[q]), 100)):
            pred_hash = ranks[q][k]
            if hash_landmark_dict[pred_hash] == hash_landmark_dict[target_hash]:
                correct += 1
        df_grouped = df_index.groupby('landmark_id').count()['id']
        target_landmark_id = hash_landmark_dict[target_hash]
        if target_landmark_id in df_grouped:
            recall_at_q = correct / min(100, df_grouped[target_landmark_id])
            recall += recall_at_q
    recall /= len(ranks)
    print('validation recall is {}'.format(recall))
    return recall


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


def matmul_sorting(q, X, k):
    score = np.matmul(q, X.T)
    rank = np.argsort(-score, axis=1)
    rank = rank[:, :k]
    return rank


def faiss_search(index_hashes_path, test_hashes_path, test_embed_path, index_embed_path, output_file_name):
    f = open(index_hashes_path, "r")
    index_hashes = [line[:-1] for line in f.readlines()[1:]]

    test_lines = list(open(test_hashes_path, 'r'))[1:]
    test_hashes = [h.strip() for h in test_lines]

    print("loading Query")
    Q_features = np.load(test_embed_path).T
    Q_features = Q_features.astype("float32")
    print(Q_features.shape)

    print("loading Index")
    X_features = np.load(index_embed_path).T
    X_features = X_features.astype("float32")
    print(X_features.shape)

    print("converting to contiguous")
    Q = np.ascontiguousarray(Q_features.T)
    X = np.ascontiguousarray(X_features.T)

    num_chunk = 1
    chunk_size = (Q.shape[0] // num_chunk) + 1

    print("Start Retrieval")
    o = open(output_file_name, "w")
    for cc in tqdm(range(num_chunk)):
        D_Q, I_Q = doRetrieval_chunk_num(Q[cc * chunk_size:(cc + 1) * chunk_size, :], X, num_chunks=1, k=20,
                                         verbose=False)
        for i in range(I_Q.shape[0]):
            o.write(test_hashes[i + cc * chunk_size] + ",")
            for j in range(I_Q.shape[1]):
                score = np.matmul(Q[i + cc * chunk_size, :], X[I_Q[i, j], :])
                o.write(index_hashes[I_Q[i,j]] + " " +  str(int(score * 1000000))  + " ")
                # o.write(index_hashes[I_Q[i, j]] + " ")
            o.write("\n")
            o.flush()
    o.close()
