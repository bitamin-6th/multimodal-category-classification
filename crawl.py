import requests
from bs4 import BeautifulSoup
import pandas as pd


### make dataset from Naver Shopping
### urls : a number of naver shopping item url [url1, url2, ...]
### example of url : https://search.shopping.naver.com/best100v2/detail.nhn?catId=50001279
### function returns DataFrame of item list
### Column : (image_url, name, review_url)

def make_dataset(urls, filename="data.csv"):
    item_names = []
    item_image_urls = []
    review_urls = []
    labels = []
    N = len(urls)
    for i in range(N):
        url = urls[i]
        res = requests.get(url)
        text = res.text
        soup = BeautifulSoup(text, 'html.parser')
        item_links = soup.find_all("a")
        for idx, link in enumerate(item_links):
            try:
                review_url = link["href"]
                item_name = link.find("img")["alt"]
                item_image_url = link.find("img")["data-original"]
                item_names.append(item_name)
                item_image_urls.append(item_image_url)
                review_urls.append(review_url)
                labels.append(label)
            except:
                continue

    df = pd.DataFrame()
    df["image_url"] = item_image_urls
    df["name"] = item_names
    df["review_url"] = review_urls
    df.to_csv(filename, index=False)
    return df

urls = ["https://search.shopping.naver.com/best100v2/detail.nhn?catId=50001279"]
make_dataset(urls)