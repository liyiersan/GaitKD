import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseLoss


class SpatialNorm(nn.Module):
    def __init__(self,divergence='kl'):
        if divergence =='kl':
            self.criterion = nn.KLDivLoss()
        else:
            self.criterion = nn.MSELoss()

        self.norm = nn.Softmax(dim=-1)
    
    def forward(self,pred_S,pred_T):
        norm_S = self.norm(pred_S)
        norm_T = self.norm(pred_T)

        loss = self.criterion(pred_S,pred_T)
        return loss


class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()
    def forward(self,featmap):
        n,c,h,w = featmap.shape
        featmap = featmap.reshape((n,c,-1))
        featmap = featmap.softmax(dim=-1)
        return featmap


class FeatMapKD_CWDLoss(BaseLoss):
    def __init__(self, s_channels, t_channels, norm_type='none',divergence='mse',temperature=1.0, loss_term_weight=1.0):
        super(FeatMapKD_CWDLoss, self).__init__(loss_term_weight)
        # define normalize function
        if norm_type == 'channel':
            self.normalize = ChannelNorm()
        elif norm_type =='spatial':
            self.normalize = nn.Softmax(dim=1)
        elif norm_type == 'channel_mean':
            self.normalize = lambda x:x.view(x.size(0),x.size(1),-1).mean(-1)
        else:
            self.normalize = None
        self.norm_type = norm_type

        # define loss function
        if divergence == 'mse':
            self.criterion = nn.MSELoss(reduction='sum')
        elif divergence == 'kl':
            self.criterion = nn.KLDivLoss(reduction='sum')
            self.temperature = temperature
        self.divergence = divergence
        self.conv = nn.Conv2d(s_channels, t_channels, kernel_size=1, bias=False)

    def forward(self,preds_S, preds_T):
        n,c,h,w = preds_S.shape
        
        if preds_S.size(1) != preds_T.size(1):
            preds_S = self.conv(preds_S)

        if self.normalize is not None:
            norm_s = self.normalize(preds_S/self.temperature)
            norm_t = self.normalize(preds_T.detach()/self.temperature)
        else:
            norm_s = preds_S[0]
            norm_t = preds_T[0].detach()
        
        if self.divergence == 'kl':
            norm_s = norm_s.log()

        loss = self.criterion(norm_s,norm_t)
        
        if self.norm_type == 'channel' or self.norm_type == 'channel_mean':
            loss /= n * c
            # loss /= n * h * w
        else:
            loss /= n * h * w

        loss = loss * (self.temperature**2) 
        
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info
    
    
class FeatMapKD_IFVLoss(BaseLoss):
    def __init__(self, classes, loss_term_weight=1.0):
        super(FeatMapKD_IFVLoss, self).__init__(loss_term_weight)
        self.num_classes = classes

    def forward(self, feat_S, feat_T, target):
        preds_T = feat_T.detach()
        size_f = (feat_S.shape[2], feat_S.shape[3])
        tar_feat_S = nn.Upsample(size_f, mode='nearest')(target.unsqueeze(1).float()).expand(feat_S.size())
        tar_feat_T = nn.Upsample(size_f, mode='nearest')(target.unsqueeze(1).float()).expand(feat_T.size())
        center_feat_S = feat_S.clone()
        center_feat_T = feat_T.clone()
        for i in range(self.num_classes):
            mask_feat_S = (tar_feat_S == i).float()
            mask_feat_T = (tar_feat_T == i).float()
            center_feat_S = (1 - mask_feat_S) * center_feat_S + mask_feat_S * ((mask_feat_S * feat_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)
            center_feat_T = (1 - mask_feat_T) * center_feat_T + mask_feat_T * ((mask_feat_T * feat_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)

        # cosinesimilarity along C
        cos = nn.CosineSimilarity(dim=1)
        pcsim_feat_S = cos(feat_S, center_feat_S)
        pcsim_feat_T = cos(feat_T, center_feat_T)

        # mseloss
        mse = nn.MSELoss()
        loss = mse(pcsim_feat_S, pcsim_feat_T)
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info
    
    
class FeatMapKD_StructuralKDLoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0):
        super(FeatMapKD_StructuralKDLoss, self).__init__(loss_term_weight)

    def pair_wise_sim_map(self, fea):
        B, C, H, W = fea.size()
        fea = fea.reshape(B, C, -1)
        fea_T = fea.transpose(1,2)
        sim_map = torch.bmm(fea_T, fea)
        return sim_map

    def forward(self, feat_S, feat_T):
        B, C, H, W = feat_S.size()

        #feat_S = feat_S.reshape(B, C, -1)
        patch_w = 2 #int(0.5 * W)
        patch_h = 2 #int(0.5 * H)
        maxpool = nn.MaxPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        feat_S = maxpool(feat_S)
        feat_T= maxpool(feat_T)

        feat_S = F.normalize(feat_S, p=2, dim=1)
        feat_T = F.normalize(feat_T, p=2, dim=1)
        
        S_sim_map = self.pair_wise_sim_map(feat_S)
        T_sim_map = self.pair_wise_sim_map(feat_T)
        B, H, W = S_sim_map.size()

        sim_err = ((S_sim_map - T_sim_map)**2)
        loss = sim_err.mean()
        
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info
