import numpy as np
import faiss
import os
import pandas as pd
import pickle
from tqdm import tqdm
import time
from multiprocessing import Process, Manager
from collections import deque
from joblib import Parallel, delayed


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

def get_image_paths():
    with open(os.path.join(BASE_PATH, 'index_image_paths.pkl'), 'rb') as f:
        index_image_paths = pickle.load(f)

    with open(os.path.join(BASE_PATH, 'query_image_paths.pkl'), 'rb') as f:
        query_image_paths = pickle.load(f)

    return index_image_paths, query_image_paths


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


def doRetrieval_chunk(Q, X, k=100, verbose=True):
    # X shape: n * d
    # split into 2 chunks
    X_length = X.shape[0]
    first_chunk = X[:X_length//3, :]
    D1, I1 = doRetrieval(Q, first_chunk, k=k, verbose=verbose)
    second_chunk = X[X_length//3: 2 * (X_length//3), :]
    D2, I2 = doRetrieval(Q, second_chunk, k=k, verbose=verbose) 
    third_chunk = X[2 * (X_length//3):, :]
    D3, I3 = doRetrieval(Q, third_chunk, k=k, verbose=verbose) 
    # shift the index
    I2 += first_chunk.shape[0] 
    I3 += first_chunk.shape[0] + second_chunk.shape[0]

    D = np.concatenate([D1, D2, D3], axis=1)
    I = np.concatenate([I1, I2, I3], axis=1)
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


if __name__ == "__main__":
    index_hashes = "index_list_ret.txt"
    f = open(index_hashes, "r")
    index_hashes = [line[:-1] for line in f.readlines()]
    test_hashes = "query_list_ret.txt"
    f = open(test_hashes, "r")
    test_hashes = [line[:-1] for line in f.readlines()]
    
    print("loading Query")
    Q_features = np.load("npy/query_lat_allnorm_5_kq5_new.npy").T  # shape d x n 
    Q_features = Q_features.astype("float32")
    print("loading Index")
    X_features = np.load("npy/index_lat_allnorm_5_kq5_new.npy").T
    X_features = X_features.astype("float32")
    
    print("converting to contiguous")
    Q = np.ascontiguousarray(Q_features.T)
    X = np.ascontiguousarray(X_features.T)
    
    kq = 100
    kx = 100
    # do initial retrieval
    D_Q, I_Q = doRetrieval_chunk(Q, X, k=kq, verbose=True)
    print("-----------------------------------------------------------")
    print(I_Q.shape)
    
    o = open("himanshu_gcn_embeddings_noqedba_after", "w")
#    o.write("id,images\n")
    for i in range(I_Q.shape[0]):
        o.write(test_hashes[i] + ",")
        for j in range(I_Q.shape[1]):
            o.write(index_hashes[I_Q[i,j]] + " ")
        o.write("\n")
        o.flush()
    #o.close()
#    
    # do for index:
    num_chunk = 10
    chunk_size = (X.shape[0] // num_chunk) + 1
    for i in tqdm(range(num_chunk)):
        D_X, I_X = doRetrieval_chunk(X[i*chunk_size: (i+1) * chunk_size, :], X, k=kq, verbose=False)
        #D, I = index.search(X[i*chunk_size: (i+1) * chunk_size, k])
        for j in range(I_X.shape[0]):
            o.write(index_hashes[j + i*chunk_size] + ",")
            for k in range(I_X.shape[1]):
                o.write(index_hashes[I_X[j,k]] + " ")
            o.write("\n")
            o.flush()



#    BASE_PATH = '/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/2019_landmark/'
#    index_csv = os.path.join(BASE_PATH, 'index_val_new.csv')
#    test_csv = os.path.join(BASE_PATH, 'query_val_new.csv')
#    
#    df_index = pd.read_csv(index_csv)
#    index_image_hashes = list(df_index['id'])
#    df_query = pd.read_csv(test_csv)
#    test_image_hashes = list(df_query['id'])
#    images, qimages = get_image_paths()
#    ranks = I_Q
#    ranks_list = []
#    for i in range(ranks.shape[0]):
#        ranks_list.append([test_image_hashes[i]])
#        for j in range(ranks.shape[1]):
#            index = ranks[i, j]
#            ranks_list[i].append(index_image_hashes[index])
#    compute_map(df_index, df_query, ranks_list)
