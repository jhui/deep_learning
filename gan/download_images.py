import urllib.request
import numpy as np
import cv2
import untangle
import os

maxsize = 512

# tags = ["asu_tora","puuakachan","mankun","hammer_%28sunset_beach%29",""]
# for tag in tags:

count = 50311
directory = "/Users/venice/dataset/anime/imgs"

if not os.path.exists(directory):
    os.makedirs(directory)

for i in range(506, 10000):
    stringreturn = urllib.request.urlopen("http://safebooru.org/index.php?page=dapi&s=post&q=index&tags=1girl%20solo&pid="+str(i)).read().decode('utf-8')
    xmlreturn = untangle.parse(stringreturn)
    for post in xmlreturn.posts.post:
        imgurl = "http:" + post["sample_url"]
        if ("png" in imgurl) or ("jpg" in imgurl):
            print(f"{i}:{count} {imgurl}")
            resp = urllib.request.urlopen(imgurl)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            height, width = image.shape[:2]
            if height > width:
                scalefactor = (maxsize*1.0) / width
                res = cv2.resize(image,(int(width * scalefactor), int(height*scalefactor)), interpolation = cv2.INTER_CUBIC)
                cropped = res[0:maxsize,0:maxsize]
            if width > height:
                scalefactor = (maxsize*1.0) / height
                res = cv2.resize(image,(int(width * scalefactor), int(height*scalefactor)), interpolation = cv2.INTER_CUBIC)
                center_x = int(round(width*scalefactor*0.5))
                print(center_x)
                cropped = res[0:maxsize,center_x - maxsize//2:center_x + maxsize//2]

            count += 1
            cv2.imwrite(f"{directory}/{str(count)}.jpg",cropped)
