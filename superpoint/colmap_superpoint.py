from extract import extract_all_features
from matchers import mutual_nearest
import cv2
import os
import numpy as np
import argparse
import sqlite3

class Database:
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        self.cursor = self.conn.cursor()
        
    def create_tables(self):
        self.cursor.executescript("""
            CREATE TABLE IF NOT EXISTS cameras (
                camera_id INTEGER PRIMARY KEY,
                model INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                params BLOB);
            
            CREATE TABLE IF NOT EXISTS images (
                image_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                camera_id INTEGER NOT NULL);
            
            CREATE TABLE IF NOT EXISTS keypoints (
                image_id INTEGER PRIMARY KEY,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB);
            
            CREATE TABLE IF NOT EXISTS descriptors (
                image_id INTEGER PRIMARY KEY,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB);
            
            CREATE TABLE IF NOT EXISTS matches (
                pair_id INTEGER PRIMARY KEY,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB);
        """)
        
    def add_camera(self, model, w, h, params, cam_id=0):
        params_blob = np.array(params, dtype=np.float64).tobytes()
        self.cursor.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?)",
            (cam_id, model, w, h, params_blob))
        
    def add_image(self, name, cam_id, img_id):
        self.cursor.execute(
            "INSERT INTO images VALUES (?, ?, ?)",
            (img_id, name, cam_id))
        
    def add_keypoints(self, img_id, kpts):
        kpts = np.array(kpts, dtype=np.float32)
        self.cursor.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (img_id, kpts.shape[0], kpts.shape[1], kpts.tobytes()))
        
    def add_descriptors(self, img_id, descs):
        descs = np.array(descs, dtype=np.uint8)
        self.cursor.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (img_id, descs.shape[0], descs.shape[1], descs.tobytes()))
        
    def add_matches(self, id1, id2, matches):
        pair_id = id1 * 2147483647 + id2
        matches = np.array(matches, dtype=np.uint32)
        self.cursor.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id, matches.shape[0], matches.shape[1], matches.tobytes()))
        
    def execute(self, query):
        self.cursor.execute(query)
        
    def commit(self):
        self.conn.commit()
        
    def close(self):
        self.conn.close()

def setup_database(db, img_path):
    files = [f for f in os.listdir(img_path) if f.endswith(('.jpg', '.png'))]
    img = cv2.imread(os.path.join(img_path, files[0]))
    h, w = img.shape[:2]
    
    f = max(w, h) * 1.2
    params = np.array([f, w/2, h/2])
    
    db.add_camera(0, w, h, params)
    
    for i, f in enumerate(files):
        db.add_image(f, 0, i)
    
    return files

def process_features(db, img_path, files, config):
    features = extract_all_features(img_path, config)
    
    db.execute("DELETE FROM keypoints;")
    db.execute("DELETE FROM descriptors;")
    db.execute("DELETE FROM matches;")
    
    for i, f in enumerate(files):
        kpts = features[f]['kp']
        descs = features[f]['desc']
        
        kpts_formatted = np.concatenate([
            kpts.astype(np.float32),
            np.ones((len(kpts), 1), dtype=np.float32),
            np.zeros((len(kpts), 1), dtype=np.float32)
        ], axis=1)
        
        db.add_keypoints(i, kpts_formatted)
        
        if len(descs) > 0:
            descs_quantized = (descs * 255).astype(np.uint8)
            db.add_descriptors(i, descs_quantized)
    
    return features

def match_images(db, features, files, match_file):
    steps = [1, 2, 3, 5, 8]
    n = len(files)
    
    with open(match_file, 'w') as f:
        for step in steps:
            for i in range(n - step):
                f.write(f"{files[i]} {files[i + step]}\n")
                
                d1 = features[files[i]]['desc']
                d2 = features[files[i + step]]['desc']
                matches = mutual_nearest(d1, d2)
                db.add_matches(i, i + step, matches)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--max_kpts", type=int, default=1000)
    args = parser.parse_args()
    
    config = {'max_keypoints': args.max_kpts}
    
    db_path = os.path.join(args.proj, "database.db")
    match_path = os.path.join(args.proj, "matches.txt")
    img_path = os.path.join(args.proj, args.images)
    
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db = Database(db_path)
    db.create_tables()
    
    files = setup_database(db, img_path)
    features = process_features(db, img_path, files, config)
    match_images(db, features, files, match_path)
    
    db.commit()
    db.close()
    
    sparse_path = os.path.join(args.proj, "sparse")
    os.makedirs(sparse_path, exist_ok=True)
    
    os.system(f"colmap matches_importer --database_path {db_path} --match_list_path {match_path} --match_type pairs")
    os.system(f"colmap mapper --database_path {db_path} --image_path {img_path} --output_path {sparse_path}")

if __name__ == "__main__":
    main()
