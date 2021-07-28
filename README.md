# nms-content-filter
Naver Multimodal Shopping Item Content Based Filtering

## 1. Dataset Curation
1. crawl.py

네이버 쇼핑 스포츠 배너에서 각 아이템별 인기 품목을 클래스별로 100개씩 수집, data.csv 생성.

2. data.csv

data.csv은 36468개의 item csv이며 (image_url, name, review_url)으로 이루어져 있음.

- image_url : 이미지의 url, 호출을 통해 이미지를 다운받을 수 있음.
- name : 상품명
- review_url : 페이지 url이며 본 url로 상품 태그나 리뷰를 재수집해야함. 

## 2. Method
이미지와 텍스트로부터 multi-modal feature를 추출함.

- Image Only
  - image_features_extraction.py 
    - 이미지 다운로드 후 ViT-B/32 를 통해 이미지 Encode -> feature extraction
    - Classification을 위한 label 생성 -> 대/중/소 분류로 다차원 label 생성

  - classification.ipynb
    - 대분류를 Label로 한 Classification (PCA 적용/ 미적용)
    - 중분류를 Label로 한 Classification (PCA 적용/ 미적용)
    - 대분류를 기준으로 부분집합 생성 후 중분류를 label로 한 Classification (PCA 적용/ 미적용)
    - cosine similarity를 이용한 Content-Based Filtering

  #|\| |with PCA| ||w.o PCA||
  |---|---|---|
  |Large||0|||0||
  |Medium||0|||0||
  |Subset||0||||0|
  
## 3. Results

