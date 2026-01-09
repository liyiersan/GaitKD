import torch
import torch.nn.functional as F
from .base import BaseLoss


class FeatsKD_MSELoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0):
        super(FeatsKD_MSELoss, self).__init__(loss_term_weight)

    def forward(self, feats_s, feats_t):
        """
        feats_s, feats_t: [N, C, P] 或 [N, C, H, W] 都可以
        """
        if torch.is_tensor(feats_s):
            feats_s = [feats_s]
        if torch.is_tensor(feats_t):
            feats_t = [feats_t]

        assert len(feats_s) == len(feats_t), \
            f"Number of feature maps mismatch: {len(feats_s)} vs {len(feats_t)}"

        loss = 0.0
        for fs, ft in zip(feats_s, feats_t):
            assert fs.shape == ft.shape, f"Shape mismatch: {fs.shape} vs {ft.shape}"
            loss = loss + F.mse_loss(fs, ft)
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info


class FeatsKD_ABLoss(BaseLoss):
    """
    "Frequency-Aligned Knowledge Distillation for Lightweight Spatiotemporal Forecasting"
    https://arxiv.org/pdf/2507.02939.pdf
    """
    
    def __init__(self, feat_num=1, margin=1.0, loss_term_weight=1.0):
        super(FeatsKD_ABLoss, self).__init__(loss_term_weight)
        self.w = [2**(i-feat_num+1) for i in range(feat_num)]
        self.margin = margin

    def forward(self, feats_s, feats_t):
        """
        feats_s, feats_t: list of tensors, e.g. [feat_s], [feat_t]
        每个 tensor: [N, C, P] 或 [N, C, H, W] 都可以
        """
        if not isinstance(feats_s, list):
            feats_s = [feats_s]
        if not isinstance(feats_t, list):
            feats_t = [feats_t]
        losses = []
        for w, s, t in zip(self.w, feats_s, feats_t):
            if len(s.shape) > 3: # [n, c, t, h, w] or [n, c, h, w]
                N = s.shape[0]
            elif len(s.shape) == 3: # [n, c, p]
                s = s.transpose(1, 2).contiguous().view(-1, s.shape[1])  # [n*p, c]
                t = t.transpose(1, 2).contiguous().view(-1, t.shape[1])  # [n*p, c]
                N = s.shape[0]
            # N = s.numel() # total number of elements
            l = self.criterion_alternative_l2(s, t)
            l = w * l / N
            losses.append(l)
        loss = torch.mean(torch.stack(losses))
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info

    def criterion_alternative_l2(self, source, target):
        loss = ((source + self.margin) ** 2 * ((source > -self.margin) & (target <= 0)).float() +
                (source - self.margin) ** 2 * ((source <= self.margin) & (target > 0)).float())
        return torch.abs(loss).sum()


class FeatsKD_ATLoss(BaseLoss):
    """
    "Paying More Attention To The Attention - Improving the Performance of CNNs via Attention Transfer"
    https://arxiv.org/pdf/1612.03928.pdf
    Attention Transfer (AT) Loss adapted for gait.
    Supports features of shape:
      - [N, C, H, W]
      - [N, C, P]
      - [N, C, T]
    """

    def __init__(self, p=2, loss_term_weight=1.0):
        super().__init__(loss_term_weight)
        self.p = p

    def _attention_map(self, feat):
        """
        feat: [N, C, ...]
        return: normalized attention map [N, *]
        """
        # 1) channel-wise aggregation: sum(|F|^p) over C
        att = feat.abs().pow(self.p).sum(dim=1)  # [N, ...]

        # 2) flatten remaining dims & normalize
        att = att.view(att.size(0), -1)
        att = F.normalize(att, p=2, dim=1)
        return att

    def forward(self, feats_t, feats_s):
        """
        feats_t, feats_s:
          - Tensor or list of Tensor
          - each Tensor: [N, C, ...]
        """
        if torch.is_tensor(feats_t):
            feats_t = [feats_t]
        if torch.is_tensor(feats_s):
            feats_s = [feats_s]

        assert len(feats_t) == len(feats_s), \
            f"Number of layers mismatch: {len(feats_t)} vs {len(feats_s)}"

        loss = 0.0
        for ft, fs in zip(feats_t, feats_s):
            assert ft.shape == fs.shape, f"Shape mismatch: {ft.shape} vs {fs.shape}"

            at_t = self._attention_map(ft)
            at_s = self._attention_map(fs)

            loss = loss + F.mse_loss(at_s, at_t)

        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info

