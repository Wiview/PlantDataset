import numpy as np
import torch

def nn_match(d1, d2):
    sim = torch.einsum('dn,dm->nm', d1, d2)
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    
    ids1 = torch.arange(0, sim.shape[0], device=d1.device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]]).t()
    
    return matches.cpu().numpy()

def mutual_nearest(d1, d2):
    if isinstance(d1, np.ndarray):
        d1 = torch.from_numpy(d1).float()
    if isinstance(d2, np.ndarray):
        d2 = torch.from_numpy(d2).float()
    
    return nn_match(d1.t(), d2.t())

def ratio_test(d1, d2, ratio=0.8):
    if isinstance(d1, np.ndarray):
        d1 = torch.from_numpy(d1).float()
    if isinstance(d2, np.ndarray):
        d2 = torch.from_numpy(d2).float()
    
    sim = torch.einsum('dn,dm->nm', d1.t(), d2.t())
    
    nns = torch.topk(sim, 2, dim=1)
    ratios = nns.values[:, 0] / (nns.values[:, 1] + 1e-8)
    
    valid = ratios > ratio
    matches = torch.stack([torch.arange(sim.shape[0])[valid], nns.indices[valid, 0]]).t()
    
    return matches.cpu().numpy()

def superglue_match(d1, d2, scores1, scores2):
    sim = torch.einsum('dn,dm->nm', d1.t(), d2.t())
    
    if isinstance(scores1, np.ndarray):
        scores1 = torch.from_numpy(scores1).float()
    if isinstance(scores2, np.ndarray):
        scores2 = torch.from_numpy(scores2).float()
    
    score_matrix = scores1[:, None] * scores2[None, :]
    sim = sim * score_matrix
    
    return nn_match(d1.t(), d2.t())
