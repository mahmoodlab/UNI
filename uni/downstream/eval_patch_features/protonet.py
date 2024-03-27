"""
Implementation adapted from: https://github.com/mbanani/lgssl/blob/df45bae647fc24dce8a6329eb697944053e9a8a0/lgssl/evaluation/fewshot.py#L9C3-L9C3
"""
from typing import Tuple

import faiss
import numpy as np
import pandas as pd
import sklearn.cluster
from threadpoolctl import threadpool_limits
import torch
from torch.nn.functional import normalize

class ProtoNet:
    """
    Sklearn-like class for SimpleShot.

    Implementation adapted from: https://github.com/mbanani/lgssl/blob/df45bae647fc24dce8a6329eb697944053e9a8a0/lgssl/evaluation/fewshot.py#L9C3-L9C3
    """
    def __init__(
            self, 
            index_type: str = 'Flat', 
            metric: str = 'L2', 
            center_feats: bool = True, 
            normalize_feats: bool = True
    ) -> None:
        self.index_type = index_type
        self.metric = metric
        self.center_feats = center_feats
        self.normalize_feats = normalize_feats
	
    def fit(
            self, 
            X: torch.Tensor, 
            y: torch.Tensor, 
            verbose: bool=True
    ) -> None:
        """
        Averages feature embeddings over each class to create "class prototypes", e.g. - one-shot support examples.

        Args:
            X (torch.Tensor): [N x D]-dim feature matrix as support to create class prototypes (N: num samples, D: feature dim)
            y (torch.Tensor): [N]-dim label vector aligned with X (N: num samples)
            verbose (bool, optional):  Defaults to False.
        """
        feats_source, labels_source = X, y

        ### Assert labels have correct shape and that # For C classes, the labels must be in [0, C-1]
        assert len(labels_source.shape) == 1
        prototype_labels = torch.Tensor(sorted(np.unique(labels_source))).to(torch.long)
        assert prototype_labels.max().item() == len(prototype_labels) - 1     # For C classes, y must be in [0, C-1]
        self.prototype_labels = prototype_labels

        if verbose:
            print('Num features averaged per class prototype:')
            for cls, count in pd.DataFrame(labels_source.numpy()).value_counts().sort_index(ascending=True).items():
                print(f'\tClass {cls[0]}: {count}')

        ### Apply centering and normalization (if set)
        if self.center_feats:
            if verbose: print("Applying centering...")
            self.mean = feats_source.mean(dim=0, keepdims=True)     # [1 x D]-dim vector, average taken over samples
            feats_source = feats_source - self.mean
        else:
            self.mean = None

        if self.normalize_feats:
            if verbose: print("Applying normalization...")
            feats_source = normalize(feats_source, dim=-1, p=2)

        ### Compute prototypes (avoiding using the one-line implementation for readability)
        # self.prototype_embeddings = torch.vstack([feats_source[np.where(labels_source == c)[0]].mean(dim=0) for c in self.prototype_labels])
        prototype_embeddings = []
        for c in self.prototype_labels:
            class_inds = np.where(labels_source == c.item())[0]
            class_feats = feats_source[class_inds].mean(dim=0)
            prototype_embeddings.append(class_feats)
        self.prototype_embeddings = torch.vstack(prototype_embeddings)
        assert self.prototype_embeddings.shape == (len(self.prototype_labels), feats_source.shape[1])
	

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Gets the closest prototype for each query in X.

        Args:
            X (torch.Tensor): [N x D]-dim query feature matrix (N: num samples, D: feature dim)

        Returns:
            labels_pred (torch.Tensor): N-dim label vector for X (labels assigned from cloest prototype for each query in X)
        """
        feats_query = X

        ### Apply centering and normalization (if set)
        if self.center_feats:
            feats_query = feats_query - self.mean

        if self.normalize_feats:
            feats_query = normalize(feats_query, dim=-1, p=2)

        ### Compute distances, and get the closest prototype for each query as the label
        feats_query = feats_query[:, None]                                  # [N x 1 x D]
        prototype_embeddings = self.prototype_embeddings[None, :]           # [1 x C x D]
        pw_dist = (feats_query - prototype_embeddings).norm(dim=-1, p=2)    # [N x C x D] distance/sim matrix
        labels_pred = self.prototype_labels[pw_dist.min(dim=1).indices]     # [N,] label vector
        return labels_pred
    

    def get_topk_queries(
            self, 
            X: torch.Tensor, 
            topk: int = 100, 
            center_feats: bool = False, 
            normalize_feats: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the top-k similar queries to each class prototype. 
        This function is used by "prototype_topk_vote", to get distance matrix for all queries to all prototypes for "top-k" voting.

        Args:
            X (torch.Tensor): [N x D]-dim feature matrix as query (N: num samples, D: feature dim)
            topk (int, optional): Number of queries to retrieve per prototype. Defaults to 100.
            center_feats (bool, optional): Whether or not these queries should be centered. Defaults to False.
            normalize_feats (bool, optional): Whether or not these queries should be normalized. Defaults to False.

        Returns:
            X_nearest (torch.Tensor): [C*topk x D]-dim feature matrix of top-k queries per prototype (C: num classes, topk: num queries, D: feature dim)
            y_nearest (torch.Tensor): (C*topk)-dim label vector aligned with X_nearest (C: num classes, topk: num queries
            dist (torch.Tensor): [C x topk]-dim distance vector aligned with X_nearest (C: num classes, topk: num queries
        """
        dist, topk_inds = self._get_topk_queries_inds(X, topk=topk) # [C x topk]-dim, [C x topk]-dim distance and index matrix
        X_nearest = torch.vstack([X[topk_inds[int(i)]] for i in self.prototype_labels])
        y_nearest = torch.Tensor(np.concatenate([[int(i),]*topk_inds.shape[1] for i in self.prototype_labels]))

        ### Apply centering and normalization (if set)
        if center_feats:
            assert self.mean is not None
            X_nearest = X_nearest - self.mean

        if normalize_feats:
            X_nearest = normalize(X_nearest, dim=-1, p=2)
        
        
        return X_nearest, y_nearest, dist
    
    
    def get_topk_queries_with_label(
            self, 
            X: torch.Tensor, 
            y: torch.Tensor, 
            topk: int = 100, 
            center_feats: bool = False, 
            normalize_feats: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the top-k similar queries to each class prototype, with the ground truth labels also sorted. Specifically, note that:
        - "y_nearest" are the "labels" of using the prototypes to retrieve the top-k queries
        - "y" are the actual labels of the queries (if known)

        Args:
            X (torch.Tensor): [N x D]-dim feature matrix as query (N: num samples, D: feature dim)
            y (torch.Tensor): N-dim ground truth label vector of the queries
            topk (int, optional): Number of queries to retrieve per prototype. Defaults to 100.
            center_feats (bool, optional): Whether or not these queries should be centered. Defaults to False.
            normalize_feats (bool, optional): Whether or not these queries should be normalized. Defaults to False.

        Returns:
            X_nearest (torch.Tensor): [C*topk x D]-dim feature matrix of top-k queries per prototype (C: num classes, topk: num queries to retrieve, D: feature dim)
            y_nearest (torch.Tensor): (C*topk)-dim label vector aligned with X_nearest (C: num classes, topk: num queries to retrieve)
            y_label (torch.Tensor): (C*topk)-dim "ground truth" label vector aligned with X_nearest (C: num classes, topk: num queries to retrieve)
            dist (torch.Tensor): [C x topk]-dim distance vector aligned with X_nearest (C: num classes, topk: num queries to retrieve)
        """
        # [C x topk]-dim, [C x topk]-dim distance and index matrix
        dist, topk_inds = self._get_topk_queries_inds(X, topk=topk) 
        X_nearest = torch.vstack([X[topk_inds[int(i)]] for i in self.prototype_labels])
        y_nearest = torch.Tensor(np.concatenate([[int(i),]*topk_inds.shape[1] for i in self.prototype_labels]))
        if y is None:
            y_label is None
        else:
            y_label = torch.cat([y[topk_inds[int(i)]] for i in self.prototype_labels])

        if center_feats:
            assert self.mean is not None
            X_nearest = X_nearest - self.mean

        if normalize_feats:
            X_nearest = normalize(X_nearest, dim=-1, p=2)
            
        return X_nearest, y_label, y_nearest, dist
        
        
    def _get_topk_queries_inds(self, X: torch.Tensor, topk=100) -> Tuple[np.array, np.array]:
        """
        Gets the distances and indices of the top-k queries to each prototype via faiss.

        Args:
            X (torch.Tensor): [N x D]-dim feature matrix as query (N: num samples, D: feature dim)
            topk (int, optional): Number of queries to retrieve per prototype. Defaults to 100.

        Returns:
            D (torch.Tensor): [C x topk]-dim distance vector aligned with X_nearest (C: num classes, topk: num queries to retrieve)
            I (torch.Tensor): [C x topk]-dim index vector aligned with X_nearest (C: num classes, topk: num queries to retrieve)
        """
        feats_query = X

        if self.center_feats:
            feats_query = feats_query - self.mean

        if self.normalize_feats:
            feats_query = normalize(feats_query, dim=-1, p=2)

        # compute distances
        if self.metric == 'L2':
            index = faiss.IndexFlatL2(feats_query.shape[1])
        else:
            index = faiss.IndexFlatIP(feats_query.shape[1])
        
        index.add(feats_query.numpy())
        D, I = index.search(self.prototype_embeddings.numpy(), topk)
        return D, I


    def _get_topk_prototypes_inds(
            self, 
            X: torch.Tensor, 
            topk: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the top-k similar class prototypes to each query. 

        Args:
            X (torch.Tensor): [N x D]-dim feature matrix as query (N: num samples, D: feature dim)
            topk (int, optional): Defaults to 1.

        Returns:
            D (np.ndarray): [N x topk]-dim distance matrix (N: num samples, topk: top-k similar prototypes to retrieve)
            I (np.ndarray): [N x topk]-dim index matrix (N: num samples, topk: top-k similar prototypes to retrieve)
        """
        feats_query = X

        if self.center_feats:
            feats_query = feats_query - self.mean

        if self.normalize_feats:
            feats_query = normalize(feats_query, dim=-1, p=2)

        # compute distances
        if self.metric == 'L2': # faiss.METRIC_L2
            index = faiss.IndexFlatL2(feats_query.shape[1])
        else:
            index = faiss.IndexFlatIP(feats_query.shape[1])
        index.add(self.prototype_embeddings.numpy())
        D, I = index.search(feats_query.numpy(), topk)
        return D, I
    

def prototype_topk_vote(
        clf, 
        X_query, 
        topk: int = 5
) -> int:
    """
    Predicts the class label of a bag of features by taking the majority vote of the topk retrieved patches.

    Args:
        clf (ProtoNet): ProtoNet object
        X_query (torch.Tensor): [N x D]-dim bag of features to predict
        topk (int): number of scores of the retrieved patches to consider for voting.

    Returns:
        (int): predicted class label
    """
    # dist is a [C x topk] matrix of distances (C: num prototypes, topk: num retrieved patches)
    _, _, dist = clf.get_topk_queries(X_query, topk=topk)  

    # average distance per prototype. index corresponds to prototype label
    dist = dist.sum(axis=1) / topk  

    # returns index of prototype with minimum distance / highest similarity as the predicted label
    if clf.metric == 'L2':  
        return np.argmin(dist)
    else:
        return np.argmax(dist)
