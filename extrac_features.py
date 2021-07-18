from PIL import Image
import clip
import torch
import urllib.request
import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')
img_url = data['image_url'].tolist()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def extract_features(n):
    image = []
    image_features = []

    for i in range(n):
        url = img_url[i]
        urllib.request.urlretrieve(url, str(i)+'test.png')
        image.append(preprocess(Image.open(
            str(i)+'test.png')).unsqueeze(0).to(device))
        with torch.no_grad():
            image_features.append(model.encode_image(image[i]))

    if len(image_features) == n:
        print('DONE')


extract_features(20)
