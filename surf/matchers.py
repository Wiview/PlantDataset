import numpy as np

def euclidean_dist(d1, d2):
    a2 = np.sum(d1**2, axis=1, keepdims=True)
    b2 = np.sum(d2**2, axis=1)
    ab = np.dot(d1, d2.T)
    dist = np.sqrt(np.maximum(a2 + b2 - 2 * ab, 0))
    return dist

def mutual_match(d1, d2):
    dist = euclidean_dist(d1, d2)
    nn12 = np.argmin(dist, axis=1)
    nn21 = np.argmin(dist, axis=0)
    
    ids = np.arange(len(d1))
    mask = ids == nn21[nn12]
    
    return np.column_stack([ids[mask], nn12[mask]])

def ratio_match(d1, d2, ratio=0.8):
    dist = euclidean_dist(d1, d2)
    sorted_idx = np.argsort(dist, axis=1)
    
    nn1_dist = dist[np.arange(len(dist)), sorted_idx[:, 0]]
    nn2_dist = dist[np.arange(len(dist)), sorted_idx[:, 1]]
    
    good = nn1_dist / (nn2_dist + 1e-8) <= ratio
    valid_ids = np.where(good)[0]
    
    return np.column_stack([valid_ids, sorted_idx[valid_ids, 0]])

def combined_match(d1, d2, ratio=0.8):
    mutual = mutual_match(d1, d2)
    dist = euclidean_dist(d1, d2)
    
    final_matches = []
    for i, j in mutual:
        dists = np.sort(dist[i, :])
        if len(dists) >= 2 and dists[0] / (dists[1] + 1e-8) <= ratio:
            final_matches.append([i, j])
    
    return np.array(final_matches) if final_matches else np.empty((0, 2))
