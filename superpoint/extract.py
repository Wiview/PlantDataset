import cv2
import torch
import numpy as np
from detector import PointNet
import os
from tqdm import tqdm

class SuperPointExtractor:
    def __init__(self, config={}):
        self.net = PointNet(config)
        self.net.eval()
        
    def extract(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img)[None, None]
        
        with torch.no_grad():
            pred = self.net({'image': img_tensor})
        
        kpts = pred['keypoints'][0].numpy()
        descs = pred['descriptors'][0].numpy().T
        scores = pred['scores'][0].numpy()
        
        return {'kp': kpts, 'desc': descs, 'scores': scores}

def extract_all_features(img_path, config={}):
    extractor = SuperPointExtractor(config)
    files = [f for f in os.listdir(img_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    features = {}
    
    for f in tqdm(files):
        img = cv2.imread(os.path.join(img_path, f))
        features[f] = extractor.extract(img)
    
    return features
