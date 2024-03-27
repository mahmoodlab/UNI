"""
Code based on sampler from @mileyan/simple_shot
Adapted from https://github.com/mbanani/lgssl/blob/df45bae647fc24dce8a6329eb697944053e9a8a0/lgssl/evaluation/fewshot.py.
"""

import logging
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import sklearn.neighbors
import torch
from torch.nn.functional import normalize
from torch.utils.data import Sampler
from tqdm import tqdm

from .metrics import get_eval_metrics


def eval_knn(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    center_feats: bool = True,
    normalize_feats: bool = True,
    average_feats: bool = True,
    n_neighbors: int = 20,
    num_workers: int = 8,
):
    """
    Evaluate K-Nearest Neighbors (KNN) algorithm for few-shot learning.
    Adapted from https://github.com/mbanani/lgssl/blob/df45bae647fc24dce8a6329eb697944053e9a8a0/lgssl/evaluation/fewshot.py.

    Args:
        train_feats (torch.Tensor): Training features.
        train_labels (torch.Tensor): Training labels.
        test_feats (torch.Tensor): Test features.
        test_labels (torch.Tensor): Test labels.
        center_feats (bool, optional): Whether to center the features. Defaults to True.
        normalize_feats (bool, optional): Whether to normalize the features. Defaults to True.
        average_feats (bool, optional): Whether to compute prototypes by averaging features. Defaults to True.
        n_neighbors (int, optional): Num neighbors to consider in KNN. Defaults to 20.
        num_workers (int, optional): Num workers for parallel processing. Defaults to 8.

    Returns:
        tuple: A tuple containing the following:
            - proto_metrics (dict): Results prototype-based evaluation.
            - proto_dump (dict): Dumped data for prototype-based evaluation.
            - knn_metrics (dict): Results KNN evaluation.
            - knn_dump (dict): Dumped data for KNN evaluation.
    """

    # Get train and test
    feats_source = train_feats
    labels_source = train_labels
    feats_query = test_feats
    labels_query = test_labels
    logging.info(f"KNN Evaluation: Train Shape {feats_source.shape}")
    logging.info(f"KNN Evaluation: Test Shape {feats_query.shape}")

    ### Centering features (for each channel dim across samples)
    if center_feats:
        feats_mean = feats_source.mean(dim=0, keepdims=True)
        feats_query = feats_query - feats_mean
        feats_source = feats_source - feats_mean

    ### Normalizing features across channel dim
    if normalize_feats:
        feats_source = normalize(feats_source, dim=-1, p=2)
        feats_query = normalize(feats_query, dim=-1, p=2)

    # Compute prototypes & assert labels are correct
    if average_feats:
        feats_proto = torch.vstack(
            [feats_source[np.where(labels_source == c)[0]].mean(dim=0) for c in sorted(np.unique(labels_source))]
        )
        labels_proto = torch.Tensor(sorted(np.unique(labels_source)))

    # SimpleShot Eval
    pw_dist = (feats_query[:, None] - feats_proto[None, :]).norm(dim=-1, p=2)
    labels_pred_proto = labels_proto[pw_dist.min(dim=1).indices]
    proto_metrics = get_eval_metrics(labels_query, labels_pred_proto, prefix="proto_")
    proto_dump = {
        "preds_all": labels_pred_proto,
        "targets_all": labels_query,
        "probs_all": None,
        "proto_feats": feats_proto.cpu().numpy(),
        "proto_mean": feats_mean.cpu().numpy(),
    }

    # KNN Eval
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=num_workers)
    labels_pred_knn = knn.fit(feats_source, labels_source).predict(feats_query)
    knn_metrics = get_eval_metrics(labels_query, labels_pred_knn, prefix=f"knn{n_neighbors}_")
    knn_dump = {
        "preds_all": labels_pred_knn,
        "targets_all": labels_query,
        "probs_all": None,
    }

    return knn_metrics, knn_dump, proto_metrics, proto_dump, 


def eval_fewshot(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    n_iter: int = 1000,
    n_way: int = -1,
    n_shot: int = 256,
    n_query: int = -1,
    center_feats: bool = True,
    normalize_feats: bool = True,
    average_feats: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Evaluate few-shot learning performance.

    Args:
        train_feats (torch.Tensor): Training features.
        train_labels (torch.Tensor): Training labels.
        test_feats (torch.Tensor): Test features.
        test_labels (torch.Tensor): Test labels.
        n_iter (int, optional): Num iterations. Defaults to 1000.
        n_way (int, optional): Num classes per few-shot task. Defaults to -1 (use all classes in test set).
        n_shot (int, optional): Num support examples per class. Defaults to 256 examples per class in train set.
        n_query (int, optional): Num query examples per class. Defaults to -1 (use all examples in test set).
        center_feats (bool, optional): Whether to center the features. Defaults to True.
        normalize_feats (bool, optional): Whether to normalize the features. Defaults to True.
        average_feats (bool, optional): Whether to average the features. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, dict]: A tuple containing the results from every few-shot episode and its mean/std.
    """
    logging.info(
        f"FS Evaluation: n_iter: {n_iter}, n_way: {n_way}, n_shot: {n_shot}, n_query: {n_query}, center_feats: {center_feats}, normalize_feats: {normalize_feats}, average_feats: {average_feats}"
    )
    logging.info(f"FS Evaluation: Train Shape {train_feats.shape}")
    logging.info(f"FS Evaluation: Test Shape {test_feats.shape}")

    if n_way == -1:
        n_way = len(np.unique(train_labels))
        assert n_way == len(np.unique(test_labels))

    if n_query == -1:
        logging.info("Using all test samples for query")

    # Set up sampler
    fewshot_sampler = FewShotEpisodeSampler(
        train_labels,
        test_labels,
        n_iter,
        n_way,
        n_shot,
        n_query,
    )

    # test model on dataset -- really more tasks than batches
    results_all = []
    n_way = n_way
    n_shot = n_shot

    for task in tqdm(fewshot_sampler):
        source, query = task

        # get train and test
        feats_source = train_feats[source]
        labels_source = train_labels[source]
        if n_query == -1:
            feats_query = test_feats.detach().clone()
            labels_query = test_labels.detach().clone()
        else:
            feats_query = test_feats[query]
            labels_query = test_labels[query]

        # center
        if center_feats:
            feats_mean = feats_source.mean(dim=0, keepdims=True)
            feats_query = feats_query - feats_mean
            feats_source = feats_source - feats_mean

        # normalize
        if normalize_feats:
            feats_source = normalize(feats_source, dim=-1, p=2)
            feats_query = normalize(feats_query, dim=-1, p=2)

        # compute prototypes & assert labels are correct
        if average_feats:
            feats_proto = feats_source.view(n_way, n_shot, -1).mean(dim=1)
            labels_proto = labels_source.view(n_way, n_shot)
            try:
                assert (labels_proto.min(dim=1).values == labels_proto.max(dim=1).values).all()
            except:
                breakpoint()
            labels_proto = labels_proto[:, 0]
        else:
            feats_proto = feats_source
            labels_proto = labels_source

        # classify to prototypes
        pw_dist = (feats_query[:, None] - feats_proto[None, :]).norm(dim=-1, p=2)

        labels_pred = labels_proto[pw_dist.min(dim=1).indices]
        results = get_eval_metrics(labels_query, labels_pred, get_report=False, prefix=f"Kw{n_shot}s_")

        results_all.append(results)

    # compute metrics for model
    results_df = pd.DataFrame(results_all)
    results_agg = dict(
        zip(
            list(results_df.columns + "_avg") + list(results_df.columns + "_std"),
            results_df.agg(["mean", "std"], axis=0).values.flatten(),
        )
    )
    return results_df, results_agg


class FewShotEpisodeSampler(Sampler):
    """
    Sampler for generating few-shot episodes for training or evaluation.

    Adapted from https://github.com/mbanani/lgssl/blob/df45bae647fc24dce8a6329eb697944053e9a8a0/lgssl/evaluation/fewshot.py.
    """

    def __init__(
        self,
        train_labels: List[int],
        test_labels: List[int],
        n_iter: int,
        n_way: int,
        n_shot: int,
        n_query: int,
    ) -> None:
        """
        Args:
            train_labels (list): List of training labels.
            test_labels (list): List of test labels.
            n_iter (int): Number of iterations (episodes) to generate.
            n_way (int): Number of classes per episode.
            n_shot (int): Number of samples per class in the support set.
            n_query (int): Number of samples per class in the query set.
        """
        self.n_iter = n_iter
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

        train_labels = np.array(train_labels)
        self.train_ind = []
        self.test_ind = []
        unique = np.unique(train_labels)
        unique = np.sort(unique)
        for i in unique:
            train_ind = np.argwhere(train_labels == i).reshape(-1)
            self.train_ind.append(train_ind)

            test_ind = np.argwhere(test_labels == i).reshape(-1)
            self.test_ind.append(test_ind)

    def __len__(self) -> int:
        return self.n_iter

    def __iter__(self) -> Tuple[Any, Any]:
        for _ in range(self.n_iter):
            batch_gallery = []
            batch_query = []
            classes = torch.randperm(len(self.train_ind))[: self.n_way]
            for c in classes:
                train_c = self.train_ind[c.item()]
                assert len(train_c) >= (self.n_shot), f"{len(train_c)} < {self.n_shot}"
                train_pos = torch.multinomial(torch.ones(len(train_c)), self.n_shot)
                batch_gallery.append(train_c[train_pos])

                test_c = self.test_ind[c.item()]
                if len(test_c) < (self.n_query):
                    logging.info(f"test class has {len(test_c)} ins. (< {self.n_query})")
                    batch_query.append(test_c)
                else:
                    test_pos = torch.multinomial(torch.ones(len(test_c)), self.n_query)
                    batch_query.append(test_c[test_pos])

            if self.n_shot == 1:
                batch_gallery = np.array(batch_gallery)
                batch_query = np.concatenate(batch_query)
            else:
                batch_gallery = np.concatenate(batch_gallery)
                batch_query = np.concatenate(batch_query)

            yield (batch_gallery, batch_query)
