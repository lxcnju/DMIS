import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse


class DMIS(nn.Module):
    """ DMIS
    """

    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.tau = args.init_tau
        self.decay = args.decay
        self.min_tau = args.min_tau

        self.encoder = nn.Sequential(
            nn.Linear(args.input_size, 32),
            nn.ReLU(),
        )

        self.ins_score_fn = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, args.n_classes),
        )

    def instance_selection(self, codes):
        scores = self.ins_score_fn(codes).squeeze(dim=-1)
        # scores = F.sigmoid(scores)

        if self.args.var_norm == "var":
            variance = torch.std(scores.detach()) ** 2
            scores = (scores - scores.mean()) / (variance + 1e-8)
        elif self.args.var_norm == "std":
            variance = torch.std(scores.detach())
            scores = (scores - scores.mean()) / (variance + 1e-8)
        else:
            pass

        indx = F.gumbel_softmax(
            scores, tau=self.tau, hard=True, eps=1e-10
        ).unsqueeze(dim=1)
        selected_x = torch.bmm(indx, codes).squeeze(dim=1)

        return selected_x, scores

    def decay_temperature(self):
        self.tau *= self.decay
        self.tau = max(self.tau, self.min_tau)

    def forward(self, xs):
        bs, n_ins = xs.shape[0], xs.shape[1]

        xs = xs.view((bs * n_ins, -1))
        xs = self.encoder(xs)

        xs = xs.view((bs, n_ins, -1))

        selected_xs, scores = self.instance_selection(xs)

        logits = self.classifier(selected_xs)

        return logits, scores
