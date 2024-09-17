import clip
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple, Optional, cast
from PIL import Image
import requests
import numpy as np

class Model:
    
    # init
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        # load  if possible
        try:
            self.db = np.load('image_features.npy')
        except:
            self.db = np.ndarray([])
        

    # populate
    def compute_image_features(self, images: List[Image.Image]) -> np.ndarray:
        
        preprocess_images = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(preprocess_images) 
            # image_features /= image_features.norm(dim=-1, keepdim=True)
        
        if self.db.size != 1:
            self.db = np.concatenate((self.db, image_features.numpy()))
        else:
            self.db = image_features.numpy()
        
        return self.db

    def compute_text_features(self, text: str) -> np.ndarray:
        
        with torch.no_grad():
            # clip tokenize?
            text_features = self.model.encode_text(clip.tokenize([text]).to(self.device))
            # text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features.numpy()
    # query
    def query(self, query_text: str, database_embeddings: np.ndarray = None) -> np.ndarray:

        if database_embeddings is None:
            database_embeddings = self.db
        
        query_embeddings = self.compute_text_features(query_text)

        sim = query_embeddings @ database_embeddings.T
        # reduce down to 1 dimension
        sim = sim[0]
        
        sorted_similarities = sorted(zip(sim, range(database_embeddings.shape[0])), key=lambda x: x[0], reverse=True)

        return sorted_similarities

    def add_to_db(self, images: List[Image.Image]) -> np.ndarray:
        db = model.compute_image_features(image_database)
        np.save('image_features.npy', db)

    

if __name__ == '__main__':
    model = Model()


    with open('image_urls.txt', 'r') as f:
        image_urls = [line.strip() for line in f.readlines()]
    
    # only put in new images and youre good
    image_database = []
    for url in image_urls:
        image_database.append(Image.open(requests.get(url, stream=True).raw))

    model.add_to_db(image_database)
    
    

