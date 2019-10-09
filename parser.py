import requests
from lxml import html
import cv2
import numpy as np
import argparse
import urllib
parser = argparse.ArgumentParser(prog = 'face_images_parser', description='Save images from yandex pictures')
parser.add_argument('-s','--search', help='String that will be forwarded to search machine', required = True)
parser.add_argument('-o','--output', help='Output folder', required = True)
parser.add_argument('-f','--face_detection', help = 'Enable face detection')
args = parser.parse_args()
search = urllib.parse.quote_plus(args.search)
counter = 0
for p in range (1,11):
    print(f'page: {p}')
    r = requests.get(f'https://yandex.ru/images/search?p={p}&text={search}&from=tabbar&rpt=image')
    page = html.fromstring(r.text)
    hrefs = page.xpath('//div[@class="page-layout__column page-layout__column_type_content"]/div/div/div[1]//a[@class="serp-item__link"]/img/@src')
    valid_hrefs = []
    for item in hrefs:
        valid_hrefs.append(f'http://{item[2:]}')
    for k,item in enumerate(valid_hrefs):
        if args.face_detection:
            r = requests.get(item)
            nparr = np.fromstring(r.content, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                # gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = faceCascade.detectMultiScale(
            img_np,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
            )
            if len(faces) > 0:
                with open(f'{args.output}/{counter}.jpeg','wb')as fp:
                    fp.write(r.content)
        else:
            with open(f'{args.output}/{counter}.jpeg','wb')as fp:
                fp.write(r.content)
        counter+=1
