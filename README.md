# nms-content-filter
Naver Multimodal Shopping Item Content Based Filtering

## 1. Dataset Curation
1. crawl.py
네이버 쇼핑 스포츠 배너에서 각 아이템별 인기 품목을 클래스별로 100개씩 수집, data.csv 생성.

2. data.csv
column은 (image_url, name, review_url)으로 이루어져 있음.
- image_url : 이미지의 url 호출을 통해 이미지를 다운받을 수 있음.
- name : 상품명
- review_url : 페이지 url이며 본 url로 상품 태그나 리뷰를 재수집해야함. 

## 2. Method
이미지와 텍스트로부터 multi-modal feature를 추출함. 

## 3. Results

