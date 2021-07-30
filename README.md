# Naver Multimodal Sports Shopping Item Category Classification
네이버 멀티모달 쇼핑 아이템 카테고리 분류 

![figure1](https://github.com/bitamin-6th/nms-content-filter/blob/main/model.png)

- tensorflow로 구현하였습니다. 
- 이미지, 텍스트, 가격으로 약 500개 클래스를 평균 76%의 성능으로 분류합니다.

## 1. Dataset Curation
1. crawl.py

네이버 쇼핑 스포츠 배너에서 각 아이템별 인기 품목을 클래스별로 100개씩 수집, data.csv 생성.

2. data.csv

data.csv은 36468개의 item csv이며 (image_url, name, review_url)으로 이루어져 있음.

- image_url : 이미지의 url, 호출을 통해 이미지를 다운받을 수 있음.
- name : 상품명
- review_url : 페이지 url이며 본 url로 상품 태그나 리뷰를 재수집해야함. 

3. preprocessed_datasets.csv
(image_feature, text_feature, price, label)로 이루어져 있음. 
- image_feature : CLIP으로 이미지 특징벡터 추출하여 열에 저장
- text_feature : 상품명을 토크나이징하여 열에 저장
- price : 가격, standardscaler로 전처리 후 입력에 넣음. 
- label : 대분류>중분류>소분류로 label을 합침, ex) "농구공>농구화>etc"

4. main.py
- epochs : 200
- batch_size : 256
- split : 0.2
- seed : 42
- lr : 1e-4



## 2. Method
이미지와 텍스트로부터 multi-modal feature를 추출
- Text + Image + Price
  - 텍스트 : CNN-LSTM
  - 이미지 : CLIP
  - Price : 입력
  - 을 concat 하여 MLP로 카테고리를 분류
- Text only
  - 상품명으로부터 Okt 명사 추출
  - Word2Vec으로 토크나이징
  - CNN-LSTM으로 카테고리 분류 
- Image Only
  - image_features_extraction.py 
    - 이미지 다운로드 후 ViT-B/32 를 통해 이미지 Encode -> feature extraction
    - Classification을 위한 label 생성 -> 대/중/소 분류로 다차원 label 생성

  - classification.ipynb
    - 대분류를 Label로 한 Classification (PCA 적용/ 미적용)
    - 중분류를 Label로 한 Classification (PCA 적용/ 미적용)
    - 대분류를 기준으로 부분집합 생성 후 중분류를 label로 한 Classification (PCA 적용/ 미적용)
    - cosine similarity를 이용한 Content-Based Filtering

  
## 3. Results
  
- Text vs Image vs Text+Image+Price
![figure2](https://github.com/bitamin-6th/nms-content-filter/blob/main/result.png)
- image classification accuracy

#|\| w.o PCA|PCA -> Split|Split -> PCA|
|------|-----|-----|-----|
|Large|66%|76%|69%|
|Medium|60%|67%|63%|
|Subset|76%|83%|84%
|Full|57%|59%|59%|



![newplot](https://user-images.githubusercontent.com/77579408/127608528-38a775ca-d015-419a-9d83-8b0c9d3e8cf5.png)

