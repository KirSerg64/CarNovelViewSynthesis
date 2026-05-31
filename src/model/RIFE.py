import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
# from model.IFNet import *
from model.IFNet_HDv3 import *
import torch.nn.functional as F
from model.loss import *
from model.laplacian import *
from model.refine import *
from model.utils_loss import _ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class RifeModel:
    def __init__(self, args, total_steps, lr=1e-6, local_rank=-1, arbitrary=False):
        self.flownet = IFNet()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=lr, weight_decay=1e-3) # use large weight decay may avoid NaN loss
        self.warmup_scheduler = LinearLR(
            self.optimG, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=args.warm_up
        )
        self.main_scheduler  = CosineAnnealingLR(self.optimG, T_max=total_steps, eta_min=lr * 0.01)
        self.scheduler = SequentialLR(
            self.optimG, 
            schedulers=[self.warmup_scheduler, self.main_scheduler], 
            milestones=[args.warm_up]
        )        
        self.epe = EPE()
        self.lap = LapLoss()
        self.sobel = SOBEL()
        self.max_depth = args.max_depth
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)
        self.loss_depth_alpha = args.loss_depth_alpha

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
            
        if rank <= 0:
            self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))))
        
    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/flownet.pkl'.format(path))

    def inference(self, img0, img1, scale=1, scale_list=None, TTA=False, timestep=0.5):
        if scale_list is None:
            scale_list = [4, 2, 1]
        for i in range(3):
            scale_list[i] = scale_list[i] * 1.0 / scale
        imgs = torch.cat((img0, img1), 1)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        if TTA == False:
            return merged[2]
        else:
            flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2
    
    def update(self, inputs, gt, ts_alpha, mul=1, training=True, flow_gt=None):
        img0 = inputs[:, :3]
        img1 = inputs[:, 3:6]
        depth_t0 = inputs[:, 6:7]
        depth_t1 = inputs[:, 7:8]
        depth_gt = inputs[:, 8:9]
        if training:
            self.train()
        else:
            self.eval()
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill, depth_pred = self.flownet(
            torch.cat((inputs, gt), 1), scale=[8, 4, 2, 1], timestep=ts_alpha, training=training)
        pred = merged[-1]

        loss_l1 = (self.lap(pred, gt)).mean()
        # loss_tea = (self.lap(merged[3], gt)).mean()# if merged_teacher is not None else 0.0
        loss_smooth = self.sobel(flow[-1], flow[-1]*0).mean()
        # loss_depth = F.l1_loss(depth_pred / self.max_depth, depth_gt / self.max_depth)

        # --- SSIM ---
        l_ssim = 1.0 - _ssim(pred, gt)
        # --- Depth-Weighted L1 ---
        # weight = 1 / (1 + depth); normalise depth to reasonable range first
        max_d = depth_gt[depth_gt > 0].quantile(0.95).clamp(min=1.0) \
            if (depth_gt > 0).any() \
            else torch.tensor(self.max_depth, device=depth_gt.device)
        d_norm = depth_gt / max_d.detach()
        w_depth = 1.0 / (1.0 + d_norm.clamp(min=0))  # (B,1,H,W)
        l_depth_l1 = (w_depth * (pred - gt).abs()).mean()

        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_smooth + 0.8 * l_ssim + l_depth_l1 * self.loss_depth_alpha  # when training RIFEm, the weight of loss_distill should be 0.005 or 0.002
            loss_G.backward()
            self.optimG.step()
            self.scheduler.step()
        else:
            flow_teacher = flow[3]
        return merged[-1], {
            'merged_tea': depth_pred,
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[-1][:, :2],
            'flow_tea': flow[0][:, :2],
            'loss_l1': loss_l1,
            'loss_ssim': l_ssim,
            'loss_tea': loss_smooth,
            'loss_distill': loss_distill,
            'loss_depth': l_depth_l1
        }
