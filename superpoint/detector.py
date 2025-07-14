import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

def nms_fast(scores, radius):
    def max_pool(x):
        return F.max_pool2d(x, kernel_size=radius*2+1, stride=1, padding=radius)
    
    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)

def filter_border(kpts, scores, border, h, w):
    mask_h = (kpts[:, 0] >= border) & (kpts[:, 0] < (h - border))
    mask_w = (kpts[:, 1] >= border) & (kpts[:, 1] < (w - border))
    mask = mask_h & mask_w
    return kpts[mask], scores[mask]

def select_top_k(kpts, scores, k):
    if k >= len(kpts):
        return kpts, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return kpts[indices], scores

def get_descriptors(kpts, descs, s=8):
    b, c, h, w = descs.shape
    kpts = kpts - s / 2 + 0.5
    kpts /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)]).to(kpts)[None]
    kpts = kpts*2 - 1
    
    descs = F.grid_sample(descs, kpts.view(b, 1, -1, 2), mode='bilinear', align_corners=True)
    descs = F.normalize(descs.reshape(b, c, -1), p=2, dim=1)
    return descs

class PointNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = {
            'descriptor_dim': 256,
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1,
            'remove_borders': 4,
            **config
        }
        
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1a = nn.Conv2d(1, 64, 3, 1, 1)
        self.conv1b = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2a = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2b = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3a = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3b = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv4a = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv4b = nn.Conv2d(128, 128, 3, 1, 1)
        
        self.convPa = nn.Conv2d(128, 256, 3, 1, 1)
        self.convPb = nn.Conv2d(256, 65, 1, 1, 0)
        
        self.convDa = nn.Conv2d(128, 256, 3, 1, 1)
        self.convDb = nn.Conv2d(256, self.config['descriptor_dim'], 1, 1, 0)
        
        self.load_state_dict(torch.load(Path(__file__).parent / 'superpoint_v1.pth'))
    
    def forward(self, data):
        x = self.relu(self.conv1a(data['image']))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = F.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = nms_fast(scores, self.config['nms_radius'])
        
        kpts = [torch.nonzero(s > self.config['keypoint_threshold']) for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, kpts)]
        
        kpts, scores = list(zip(*[
            filter_border(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(kpts, scores)]))
        
        if self.config['max_keypoints'] >= 0:
            kpts, scores = list(zip(*[
                select_top_k(k, s, self.config['max_keypoints'])
                for k, s in zip(kpts, scores)]))
        
        kpts = [torch.flip(k, [1]).float() for k in kpts]
        
        cDa = self.relu(self.convDa(x))
        descs = self.convDb(cDa)
        descs = F.normalize(descs, p=2, dim=1)
        
        descs = [get_descriptors(k[None], d[None], 8)[0] for k, d in zip(kpts, descs)]
        
        return {'keypoints': kpts, 'scores': scores, 'descriptors': descs}
