import cv2
import os
import numpy as np
from tqdm import tqdm

class SURFDetector:
    def __init__(self, threshold=400, octaves=4, layers=3, extended=False):
        self.surf = cv2.xfeatures2d.SURF_create(
            hessianThreshold=threshold,
            nOctaves=octaves,
            nOctaveLayers=layers,
            extended=extended
        )
        self.extended = extended
    
    def __call__(self, img):
        kp, desc = self.surf.detectAndCompute(img, None)
        if kp is None:
            size = 128 if self.extended else 64
            return {'kp': np.empty((0, 2)), 'desc': np.empty((0, size))}
        
        pts = np.array([[k.pt[0], k.pt[1]] for k in kp], dtype=np.float32)
        desc = desc / np.linalg.norm(desc, axis=1, keepdims=True)
        
        return {'kp': pts, 'desc': desc}

def extract_features(img_path, threshold=400):
    files = [f for f in os.listdir(img_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    detector = SURFDetector(threshold=threshold)
    features = {}
    
    for f in tqdm(files):
        img = cv2.imread(os.path.join(img_path, f), 0)
        features[f] = detector(img)
    
    return features
