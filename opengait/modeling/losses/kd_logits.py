import torch.nn.functional as F
import torch
from .base import BaseLoss


class LogitsKD_KLLoss(BaseLoss):
    """
    分类任务用的 logits 蒸馏（KD）Loss，适配 gait 的 [N, C, P] 形式。
    用 KL-div + 温度 T，在类别维做 softmax。
    """
    def __init__(self, T=4.0, scale=2**4, loss_term_weight=1.0):
        super(LogitsKD_KLLoss, self).__init__(loss_term_weight)
        self.T = T
        self.scale = scale  # 和 CrossEntropyLoss 一样保留一个缩放系数

    def forward(self, logits_s, logits_t):
        """
        logits_s: [N, C, P]，student 的 logits
        logits_t: [N, C, P]，teacher 的 logits
        """
        n, c, p = logits_s.size()
        # 确保浮点类型一致
        logits_s = logits_s.float() * self.scale
        logits_t = logits_t.float() * self.scale

        # 在类别维做 softmax / log_softmax，保留 part 维 P
        # 形状仍然是 [N, C, P]
        log_p_s = F.log_softmax(logits_s / self.T, dim=1)
        p_t     = F.softmax(logits_t / self.T, dim=1)

        # KL-div：
        loss = F.kl_div(log_p_s, p_t, reduction='batchmean') * (self.T * self.T) / p

        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info



class LogitsKD_KA_PSLoss(BaseLoss):
    """
    KA + Probability Shift (PS) for logits distillation, for gait logits [N, C, P].
    If teacher predicts wrong, swap prob of predicted class and true class (on soft targets).
    """
    def __init__(self, T=4.0, scale=2**4, loss_term_weight=1.0):
        super().__init__(loss_term_weight)
        self.T = T
        self.scale = scale

    @torch.no_grad()
    def _probability_shift(self, p_t, labels):
        """
        p_t: [N, C, P] teacher soft probs (already softmaxed)
        labels: [N] ground truth class indices
        """
        N, C, P = p_t.shape
        # teacher top-1 class per (N,P)
        pred = p_t.argmax(dim=1)  # [N, P]
        labels_np = labels.view(N, 1).expand(N, P)  # [N, P]

        wrong = (pred != labels_np)  # [N, P] mask

        if wrong.any():
            n_idx, p_idx = wrong.nonzero(as_tuple=True)          # indices where teacher is wrong
            pred_cls = pred[n_idx, p_idx]                        # predicted class indices
            true_cls = labels_np[n_idx, p_idx]                   # true class indices

            tmp = p_t[n_idx, pred_cls, p_idx].clone()
            p_t[n_idx, pred_cls, p_idx] = p_t[n_idx, true_cls, p_idx]
            p_t[n_idx, true_cls, p_idx] = tmp

        return p_t

    def forward(self, logits_s, logits_t, labels):
        """
        logits_s: [N, C, P]
        logits_t: [N, C, P]
        labels:   [N]
        """
        logits_s = logits_s.float() * self.scale
        logits_t = logits_t.float() * self.scale

        log_p_s = F.log_softmax(logits_s / self.T, dim=1)

        with torch.no_grad():
            p_t = F.softmax(logits_t / self.T, dim=1)            # [N, C, P]
            p_t = self._probability_shift(p_t, labels)           # KA-PS 校正

        # KL(teacher || student) in PyTorch form: kl_div(input=log_p_s, target=p_t)
        loss = F.kl_div(log_p_s, p_t, reduction="batchmean") * (self.T * self.T) / logits_s.size(-1)

        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info


class LogitsKD_MSELoss(BaseLoss):
    def __init__(self, scale=2**4, loss_term_weight=1.0):
        super(LogitsKD_MSELoss, self).__init__(loss_term_weight)
        self.scale = scale

    def forward(self, logits_s, logits_t):
        """
        logits_s, logits_t: [N, C, P]
        直接在 logits 上做 MSE Loss
        """
        if torch.is_tensor(logits_s):
            logits_s = [logits_s]
        if torch.is_tensor(logits_t):
            logits_t = [logits_t]

        assert len(logits_s) == len(logits_t), \
            f"Number of feature maps mismatch: {len(logits_s)} vs {len(logits_t)}"

        loss = 0.0
        for ls, lt in zip(logits_s, logits_t):
            assert ls.shape == lt.shape, f"Shape mismatch: {ls.shape} vs {lt.shape}"
            ls = ls.float() * self.scale
            lt = lt.float() * self.scale
            loss = loss + F.mse_loss(ls, lt)
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info


class LogitsKDLoss_KA_LSR(BaseLoss):
    """
    KA + LSR replacement for logits distillation, for gait logits [N, C, P].
    If teacher predicts wrong, replace teacher soft target with smoothed one-hot distribution.
    """
    def __init__(self, T=4.0, scale=2**4, eps=0.1, loss_term_weight=1.0):
        super().__init__(loss_term_weight)
        self.T = T
        self.scale = scale
        self.eps = eps  # smoothing strength

    @torch.no_grad()
    def _lsr_distribution(self, labels, C, P, device):
        """
        build LSR dist for labels: [N] -> [N, C, P]
        LSR: true class prob = 1-eps, others = eps/(C-1)
        """
        N = labels.size(0)
        p = torch.full((N, C, P), self.eps / (C - 1), device=device)
        p.scatter_(1, labels.view(N, 1, 1).expand(N, 1, P), 1.0 - self.eps)
        return p

    def forward(self, logits_s, logits_t, labels):
        """
        logits_s: [N, C, P]
        logits_t: [N, C, P]
        labels:   [N]
        """
        logits_s = logits_s.float() * self.scale
        logits_t = logits_t.float() * self.scale

        N, C, P = logits_s.shape
        log_p_s = F.log_softmax(logits_s / self.T, dim=1)

        with torch.no_grad():
            p_t = F.softmax(logits_t / self.T, dim=1)          # [N, C, P]
            pred = p_t.argmax(dim=1)                           # [N, P]
            labels_np = labels.view(N, 1).expand(N, P)         # [N, P]
            wrong = (pred != labels_np)                        # [N, P]

            if wrong.any():
                lsr = self._lsr_distribution(labels, C, P, device=logits_s.device)  # [N,C,P]
                # 对 wrong 的 (n,p) 位置，用 lsr[:, :, p] 替换 p_t[:, :, p]
                n_idx, p_idx = wrong.nonzero(as_tuple=True)
                p_t[n_idx, :, p_idx] = lsr[n_idx, :, p_idx]

        loss = F.kl_div(log_p_s, p_t, reduction="batchmean") * (self.T * self.T) / P

        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info



def _get_gt_mask(logits_2d, target_1d):
    # logits_2d: [B, C], target_1d: [B]
    return torch.zeros_like(logits_2d).scatter_(1, target_1d.unsqueeze(1), 1).bool()


def _cat_mask(prob, gt_mask, other_mask):
    # prob: [B, C]
    # -> [B, 2] : [P(gt), P(others_sum)]
    p_gt = (prob * gt_mask).sum(dim=1, keepdim=True)
    p_other = (prob * other_mask).sum(dim=1, keepdim=True)
    return torch.cat([p_gt, p_other], dim=1)


class LogitsDKDLoss(BaseLoss):
    """
    DKD loss (Decoupled KD) for gait logits [N, C, P].

    返回：loss_dkd（不含 CE）
    你可以在外部自己组合： total = ce + w * dkd
    """

    def __init__(self, alpha=1.0, beta=1.0, T=4.0, scale=2**4,
                 loss_term_weight=1.0, eps=1e-12):
        super(LogitsDKDLoss, self).__init__(loss_term_weight)
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.scale = scale
        self.eps = eps

    def forward(self, logits_s, logits_t, labels):
        """
        logits_s: [N, C, P]
        logits_t: [N, C, P]
        labels:   [N]
        """
        assert logits_s.dim() == 3 and logits_t.dim() == 3, "Expect [N, C, P]"
        n, c, p = logits_s.shape
        assert logits_t.shape == (n, c, p)
        assert labels.shape[0] == n

        # 和你 CE/KD 一致：先 scale
        logits_s = logits_s.float() * self.scale
        logits_t = logits_t.float() * self.scale

        # 把 part 维拉到 batch： [N, C, P] -> [N*P, C]
        # (permute后 contiguous 再 view，避免内存不连续坑)
        s = logits_s.permute(0, 2, 1).contiguous().view(n * p, c)
        t = logits_t.permute(0, 2, 1).contiguous().view(n * p, c)

        # labels: [N] -> [N*P]
        y = labels.view(n, 1).repeat(1, p).view(-1).long().to(s.device)

        # masks
        gt_mask = _get_gt_mask(s, y)            # [B, C]
        other_mask = ~gt_mask                   # [B, C]

        # ====== TCKD: [P(gt), P(others)] 的二分类蒸馏 ======
        ps = F.softmax(s / self.T, dim=1)       # [B, C]
        pt = F.softmax(t / self.T, dim=1)       # [B, C]

        ps_2 = _cat_mask(ps, gt_mask, other_mask)  # [B, 2]
        pt_2 = _cat_mask(pt, gt_mask, other_mask)  # [B, 2]

        # 用 log(ps_2) 做 KL，避免 log(0)
        log_ps_2 = torch.log(ps_2.clamp_min(self.eps))
        tckd = F.kl_div(log_ps_2, pt_2, reduction='batchmean') * (self.T ** 2)

        # ====== NCKD: 屏蔽 GT 后，只在非GT类别上对齐 ======
        # 这里用一个很大的负数近似 -inf
        huge = 1000.0
        t_part2 = F.softmax(t / self.T - huge * gt_mask.float(), dim=1)      # [B, C]
        log_s_part2 = F.log_softmax(s / self.T - huge * gt_mask.float(), dim=1)

        nckd = F.kl_div(log_s_part2, t_part2, reduction='batchmean') * (self.T ** 2)

        loss = self.alpha * tckd + self.beta * nckd

        self.info.update({
            'loss': loss.detach().clone(),
            'tckd': tckd.detach().clone(),
            'nckd': nckd.detach().clone(),
        })
        return loss, self.info
