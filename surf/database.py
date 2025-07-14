import sqlite3
import numpy as np

class COLMAPDatabase:
    @staticmethod
    def connect(path):
        return COLMAPDatabase(path)
    
    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.cursor = self.connection.cursor()
        
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
                camera_id INTEGER NOT NULL,
                prior_qw REAL,
                prior_qx REAL,
                prior_qy REAL,
                prior_qz REAL,
                prior_tx REAL,
                prior_ty REAL,
                prior_tz REAL);
            
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
        
    def add_camera(self, model, width, height, params, camera_id=None):
        if camera_id is None:
            camera_id = len(self.cursor.execute("SELECT * FROM cameras").fetchall())
        
        params_blob = np.array(params, dtype=np.float64).tobytes()
        self.cursor.execute(
            "INSERT INTO cameras (camera_id, model, width, height, params) VALUES (?, ?, ?, ?, ?)",
            (camera_id, model, width, height, params_blob))
        
    def add_image(self, name, camera_id, image_id=None):
        if image_id is None:
            image_id = len(self.cursor.execute("SELECT * FROM images").fetchall())
        
        self.cursor.execute(
            "INSERT INTO images (image_id, name, camera_id) VALUES (?, ?, ?)",
            (image_id, name, camera_id))
        
    def add_keypoints(self, image_id, keypoints):
        keypoints = np.array(keypoints, dtype=np.float32)
        self.cursor.execute(
            "INSERT INTO keypoints (image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            (image_id, keypoints.shape[0], keypoints.shape[1], keypoints.tobytes()))
        
    def add_descriptors(self, image_id, descriptors):
        descriptors = np.array(descriptors, dtype=np.uint8)
        self.cursor.execute(
            "INSERT INTO descriptors (image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            (image_id, descriptors.shape[0], descriptors.shape[1], descriptors.tobytes()))
        
    def add_matches(self, image_id1, image_id2, matches):
        pair_id = image_id1 * 2147483647 + image_id2
        matches = np.array(matches, dtype=np.uint32)
        self.cursor.execute(
            "INSERT INTO matches (pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            (pair_id, matches.shape[0], matches.shape[1], matches.tobytes()))
        
    def execute(self, query):
        self.cursor.execute(query)
        
    def commit(self):
        self.connection.commit()
        
    def close(self):
        self.connection.close()
