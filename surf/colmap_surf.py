from detector import extract_features
from matchers import mutual_match
from database import COLMAPDatabase
import cv2
import os
import numpy as np
import argparse

CAM_MODELS = {
    'SIMPLE_PINHOLE': 0, 'PINHOLE': 1, 'SIMPLE_RADIAL': 2,
    'RADIAL': 3, 'OPENCV': 4, 'FULL_OPENCV': 5
}

def get_cam_params(w, h, model_id):
    f = max(w, h) * 1.2
    cx, cy = w / 2.0, h / 2.0
    if model_id == 0:
        return np.array([f, cx, cy])
    elif model_id == 1:
        return np.array([f, f, cx, cy])
    elif model_id == 2:
        return np.array([f, cx, cy, 0.0])
    return np.array([f, f, cx, cy, 0.0, 0.0, 0.0, 0.0])

def init_db(db, img_path, cam_type):
    files = [f for f in os.listdir(img_path) if f.endswith(('.jpg', '.png'))]
    
    img = cv2.imread(os.path.join(img_path, files[0]))
    h, w = img.shape[:2]
    
    model_id = CAM_MODELS[cam_type]
    params = get_cam_params(w, h, model_id)
    
    db.add_camera(model_id, w, h, params, camera_id=0)
    
    for i, f in enumerate(files):
        db.add_image(f, 0, image_id=i)
    
    return files

def add_features(db, img_path, files):
    features = extract_features(img_path)
    db.execute("DELETE FROM keypoints;")
    db.execute("DELETE FROM descriptors;")
    db.execute("DELETE FROM matches;")
    
    for i, f in enumerate(files):
        kpts = features[f]['kp']
        descs = features[f]['desc']
        
        kpts_formatted = np.concatenate([
            kpts.astype(np.float32),
            np.ones((len(kpts), 1)).astype(np.float32),
            np.zeros((len(kpts), 1)).astype(np.float32)
        ], axis=1)
        
        db.add_keypoints(i, kpts_formatted)
        
        if len(descs) > 0:
            descs_norm = ((descs + 1) * 127.5).astype(np.uint8)
            db.add_descriptors(i, descs_norm)
    
    return features

def match_features(db, features, files, match_file):
    steps = [1, 2, 3, 5, 8, 13, 21]
    n_imgs = len(files)
    
    with open(match_file, 'w') as f:
        for step in steps:
            for i in range(n_imgs - step):
                f.write(f"{files[i]} {files[i + step]}\n")
                
                d1 = features[files[i]]['desc']
                d2 = features[files[i + step]]['desc']
                matches = mutual_match(d1, d2).astype(np.uint32)
                db.add_matches(i, i + step, matches)

def run_colmap(cmd):
    print(f"Running: {cmd}")
    os.system(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--camera", default="SIMPLE_RADIAL")
    args = parser.parse_args()
    
    db_path = os.path.join(args.proj, "database.db")
    match_path = os.path.join(args.proj, "matches.txt")
    img_path = os.path.join(args.proj, args.images)
    
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db = COLMAPDatabase.connect(db_path)
    db.create_tables()
    
    files = init_db(db, img_path, args.camera)
    features = add_features(db, img_path, files)
    match_features(db, features, files, match_path)
    
    db.commit()
    db.close()
    
    sparse_path = os.path.join(args.proj, "sparse")
    os.makedirs(sparse_path, exist_ok=True)
    
    run_colmap(f"colmap matches_importer --database_path {db_path} --match_list_path {match_path} --match_type pairs")
    run_colmap(f"colmap mapper --database_path {db_path} --image_path {img_path} --output_path {sparse_path}")

if __name__ == "__main__":
    main()
