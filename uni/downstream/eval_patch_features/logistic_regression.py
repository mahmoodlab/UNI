"""
based on https://visualstudiomagazine.com/articles/2021/06/23/logistic-regression-pytorch.aspx
"""

import numpy as np
import torch


class LogisticRegression:
    def __init__(self, C, max_iter, verbose, random_state, **kwargs):
        self.C = C
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.max_iter = max_iter
        self.random_state = random_state
        self.logreg = None
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_loss(self, feats, labels):
        loss = self.loss_func(feats, labels)
        wreg = 0.5 * self.logreg.weight.norm(p=2)
        return loss.mean() + (1.0 / self.C) * wreg

    def predict_proba(self, feats):
        assert self.logreg is not None, "Need to fit first before predicting probs"
        return self.logreg(feats.to(self.device)).softmax(dim=-1)

    def fit(self, feats, labels):
        feat_dim = feats.shape[1]
        num_classes = len(torch.unique(labels))

        # set random seed
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.logreg = torch.nn.Linear(feat_dim, num_classes, bias=True)
        self.logreg.weight.data.fill_(0.0)
        self.logreg.bias.data.fill_(0.0)

        # move everything to CUDA .. otherwise why are we even doing this?!
        self.logreg = self.logreg.to(self.device)
        feats = feats.to(self.device)
        labels = labels.to(self.device)

        # define the optimizer
        opt = torch.optim.LBFGS(
            self.logreg.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=self.max_iter,
        )
        if self.verbose:
            pred = self.logreg(feats)
            loss = self.compute_loss(pred, labels)
            print(f"(Before Training) Loss: {loss:.3f}")

        def loss_closure():
            opt.zero_grad()
            pred = self.logreg(feats)
            loss = self.compute_loss(pred, labels)
            loss.backward()
            return loss

        opt.step(loss_closure)  # get loss, use to update wts

        if self.verbose:
            pred = self.logreg(feats)
            loss = self.compute_loss(pred, labels)
            print(f"(After Training) Loss: {loss:.3f}")