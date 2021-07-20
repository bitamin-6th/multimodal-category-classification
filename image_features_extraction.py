from PIL import Image
import clip
import torch
import urllib.request
import pandas as pd
import numpy as np
import os
import csv

data = pd.read_csv('data.csv')
img_url = data['img_url'].tolist()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = []
image_features = []
error = []
path = 'images_data/'

if not os.path.isdir(path):
    os.makedirs(path)


def save_image(n, link):
    url = img_url[n]
    urllib.request.urlretrieve(url, link)


def preprocess_image(link):
    image.append(preprocess(Image.open(link)).unsqueeze(0).to(device))


def encode_image(N):
    for n in range(N):
        file = 'test_'+str(n)+'.png'
        try:
            save_image(n, file+path)
            preprocess_image(path+file)
        except:
            error.append(str(n))

        with torch.no_grad():
            image_features.append(model.encode_image(
                image[n]).numpy().reshape(-1))

    print(len(image), len(error), len(image_features))

    if (len(image)+len(error)) == len(img_url) and len(image) == len(image_features):
        print('Well Done')
    else:
        print('Something is going wrong')

    return image, error, image_features


index_ = []


def concat_dataset(origin, image_features):
    origin_data = pd.read_csv(origin)
    origin_data = pd.DataFrame(origin_data)
    origin_data = origin_data['image_url']

    for idx, value in enumerate(error):
        index_.append(int(value))

    origin_data.drop(index=index_)

    image_data = pd.concat([origin_data, image_features], axis=1)
    image_data.to_csv('image.csv', header=True)

    return image_data


concat_dataset('data.csv', image_features)
