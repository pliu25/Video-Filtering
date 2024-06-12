import cv2
import os
from os import listdir
import numpy as np 

def save_frame_range_sec(video_path, start_sec, stop_sec, step_sec,
                         dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_inv = 1 / fps

    sec = start_sec
    while sec < stop_sec:
        n = round(fps * sec)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(
                '{}_{}_{:.2f}.{}'.format(
                    base_path, str(n).zfill(digit), n * fps_inv, ext
                ),
                frame
            )
        else:
            return
        sec += step_sec

save_frame_range_sec('exvideo.mp4',
                     0, 11, 1/24,
                     'images', 'sample_video_img')

"""def pixelate_rgb(img, window):
    n, m, _ = img.shape
    print(img.shape)
    n, m = n - n % window, m - m % window
    img1 = np.zeros((n, m, 3))
    for x in range(0, n, window):
        for y in range(0, m, window):
            img1[x:x+w2indow,y:y+window] = img[x:x+window,y:y+window].mean(axis=(0,1))
    return img1
"""

def pixelate(img: list) -> list:
    for r in range(1, len(img), 2):
        row = img[r]
        for c in range(1, len(row), 2):
 
            blue = 0
            green = 0
            red = 0
            for pixel_r in range(r-1, r+1):
                for pixel_c in range(c-1, c+1):
                    pixel = img[r][c]
                    blue+=pixel[pixel_r][0]
                    blue+=pixel[pixel_c][0]
                    green+=pixel[pixel_r][pixel_c][1]
                    red+=pixel[pixel_r][pixel_c][2]
            img[r][c][0] = blue/9 
            img[r][c][1] = green/9 
            img[r][c][2] = red/9 
    return img
cv2.imshow(pixelate("/Users/pearl.liu25/Downloads/Video-Filtering-1/images/sample_video_img_000_0.00.jpg"))


inPath = "images"
outPath = "filtered_images"
"""
for imagePath in os.listdir(inPath):
    with open(os.path.join(inPath, imagePath)) as f:
        print(cv2.imread(f"images/{imagePath}"))
        for imagePath in cv2.imread(f"images/{imagePath}"):
            #pixelate_rgb(imagePath, 20)
            print(imagePath)
            #go through every nine pixel boxes 
"""


#https://stackoverflow.com/questions/47143332/how-to-pixelate-a-square-image-to-256-big-pixels-with-python

   



    


