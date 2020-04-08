
import os
import shutil
import zipfile
import requests

if __name__ == "__main__":

    url_image = "http://images.cocodataset.org/zips/train2017.zip"
    url_anno = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    os.makedirs("data", exist_ok=True)

    try:
        r = requests.get(url_image, stream=True)
        print("Downloading train2017.zip")
        with open('train2017.zip', mode='wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
        print("Downloaded")
    except requests.exceptions.RequestException as err:
        print(err)

    try:
        r = requests.get(url_anno, stream=True)
        print("Downloading annotations_trainval2017.zip")
        with open('annotations_trainval2017.zip', mode='wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
        print("Downloaded")
    except requests.exceptions.RequestException as err:
        print(err)

    with zipfile.ZipFile("train2017.zip") as zip_file:
        print("Extracting train2017.zip")
        zip_file.extractall("data")

    with zipfile.ZipFile("annotations_trainval2017.zip") as zip_file:
        print("Extracting annotations_trainval2017.zip")
        zip_file.extractall("data")

    shutil.copy("data/annotations/instances_train2017.json", "data")
    shutil.rmtree("data/annotations")
    os.remove("train2017.zip")
    os.remove("annotations_trainval2017.zip")
    print("Completed")