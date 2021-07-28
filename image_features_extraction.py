from PIL import Image
import clip
import torch
import urllib.request
import pandas as pd
import numpy as np
import os
import csv

data = pd.read_csv('text.csv')
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
            image.append(str(n))

        try:
            with torch.no_grad():
                image_features.append(model.encode_image(
                    image[n]).numpy().reshape(-1))
        except:
            image_features.append(str(n))

    print(len(image), len(error), len(image_features))

    if (len(image)+len(error)) == len(img_url) and len(image) == len(image_features):
        print('Well Done')
    else:
        print('Something is going wrong')

    return image, error, image_features


encode_image(36468)

img_data = pd.DataFrame(image_features)
text = pd.read_csv('text.csv', lineterminator='\n')

image_data = pd.concat(['text', 'img_data'], axis=1)
image_data.fillna(0, inplace=True)

image_data['big'] = 0
image_data['medium'] = 0
image_data['small'] = 0

image_data['label'] = image_data['label'].str.split('>')

for i in range(36468):
    for j in range(3):
        if j == 0:
            image_data['big'][i] = image_data['label'][i][j]
        elif j == 1:
            image_data['medium'][i] = image_data['label'][i][j]
        elif j == 2:
            image_data['small'][i] = image_data['label'][i][j]

lst1 = ['29', '여가/생활편의', '묻고 답하기']
lst2 = ['설명보기', '묻고 답하기', '163', '13', '619', '49', '15', '200', '42',
        '586', '297', '351', '38', '최대 5% 적립, 무료 시작', '공지사항', '정보 수정요청', '전체상품']
lst3 = ['최대 5% 적립, 무료 시작', '레이어 팝업 닫기', '설명보기', '정보 수정요청', '전체상품', '30', '8', '3-4인용', '1-2인용', '29', '4-5인용', '9인용 이상', '31', '7',
        '198', '382', '5', '최저가 사러가기', '7-8인용', '6-7인용', '6', '17', '2-3인용', '1,624', '16', '공지사항', '묻고 답하기', '충전포인트로 결제시', '첫 화면으로', '5-6인용']

for i in range(36468):
    for j in range(len(lst1)):
        if data['big'][i] == lst1[j]:
            data['big'][i] = 'etc'
    for j in range(len(lst2)):
        if data['medium'][i] == lst2[j]:
            data['medium'][i] = 'etc'
    for j in range(len(lst3)):
        if data['small'][i] == lst3[j]:
            data['small'][i] = 'etc'

data.to_csv('final.csv')
