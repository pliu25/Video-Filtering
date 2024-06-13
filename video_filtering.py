import cv2
import os
from os import listdir
import numpy as np 
import math


mod_images = 'filtered_images'
video_name = 'filtered_video.mp4'
#print(os.listdir(mod_images))
# mod_images_list = []
# for image in os.listdir(mod_images):
#     mod_images_list.append(image)
#print(mod_images_list)
dir = os.listdir(mod_images)
dir.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
print(dir)

images = [img for img in dir if img.endswith(".jpg")]
print("images", images)
frame = cv2.imread(os.path.join(mod_images, images[0]))
#print(frame)
size = frame.shape[1], frame.shape[0]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, 24, size)

for image in images:
    video.write(cv2.imread(os.path.join(mod_images, image)))
    print(cv2.imread(os.path.join(mod_images, image)))


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

save_frame_range_sec('original_video.mp4',
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
    for r in range(1, len(img), 3):
        row = img[r]
        for c in range(1, len(row), 3):
            blue = 0
            green = 0
            red = 0
            pixel = img[r][c]
            blue+=img[r][c][0]
            blue+=img[r+1][c][0]
            blue+=img[r-1][c][0]
            blue+=img[r][c-1][0]
            blue+=img[r][c+1][0]
            blue+=img[r-1][c-1][0]
            blue+=img[r-1][c+1][0]
            blue+=img[r+1][c-1][0]
            blue+=img[r+1][c+1][0]
            green+=img[r][c][1]
            green+=img[r+1][c][1]
            green+=img[r-1][c][1]
            green+=img[r][c-1][1]
            green+=img[r][c+1][1]
            green+=img[r-1][c-1][1]
            green+=img[r-1][c+1][1]
            green+=img[r+1][c-1][1]
            green+=img[r+1][c+1][1]
            red+=img[r][c][2]
            red+=img[r+1][c][2]
            red+=img[r-1][c][2]
            red+=img[r][c-1][2]
            red+=img[r][c+1][2]
            red+=img[r-1][c-1][2]
            red+=img[r-1][c+1][2]
            red+=img[r+1][c-1][2]
            red+=img[r+1][c+1][2]
            img[r][c][0] = math.trunc(blue/9)
            img[r+1][c][0] = math.trunc(blue/9)
            img[r-1][c][0] = math.trunc(blue/9) 
            img[r][c-1][0] = math.trunc(blue/9) 
            img[r][c+1][0] = math.trunc(blue/9)
            img[r-1][c-1][0] = math.trunc(blue/9)
            img[r-1][c+1][0] = math.trunc(blue/9)
            img[r+1][c-1][0] = math.trunc(blue/9) 
            img[r+1][c+1][0] = math.trunc(blue/9)
            img[r][c][1] = math.trunc(green/9)
            img[r+1][c][1] = math.trunc(green/9)
            img[r-1][c][1] = math.trunc(green/9)
            img[r][c-1][1] = math.trunc(green/9)
            img[r][c+1][1] = math.trunc(green/9)
            img[r-1][c-1][1] = math.trunc(green/9)
            img[r-1][c+1][1] = math.trunc(green/9)
            img[r+1][c-1][1] = math.trunc(green/9)
            img[r+1][c+1][1] = math.trunc(green/9)
            img[r][c][2] = math.trunc(red/9) 
            img[r+1][c][2] = math.trunc(red/9)
            img[r-1][c][2] = math.trunc(red/9)
            img[r][c-1][2] = math.trunc(red/9)
            img[r][c+1][2] = math.trunc(red/9)
            img[r-1][c-1][2] = math.trunc(red/9)
            img[r-1][c+1][2] = math.trunc(red/9)
            img[r+1][c-1][2] = math.trunc(red/9)
            img[r+1][c+1][2] = math.trunc(red/9)
            #print([img[r][c][0], img[r][c][1], img[r][c][2]])
            #for pixel_r in range(r-1, r+1):
                #for pixel_c in range(c-1, c+1):
                    #pixel = img[r][c]
                    #print("pixel_r", pixel_r)
                    #print("pixel_c", pixel_c)
                    #print("pixel", pixel)
                    #blue+=pixel[pixel_r][0]
                    #blue+=pixel[pixel_c][0]
                    #green+=pixel[pixel_r][pixel_c][1]
                    #red+=pixel[pixel_r][pixel_c][2]
            #img[r][c][0] = blue/9 #math.trunc
            #img[r][c][1] = green/9 
            #img[r][c][2] = red/9 
    #print("img", img)
    return img
#cv2.imwrite(cv2.imread(f"images/sample_video_img_000_0.00.jpg"), pixelate(cv2.imread(f"images/sample_video_img_000_0.00.jpg")))
#print(pixelate(cv2.imread(f"images/sample_video_img_000_0.00.jpg")))
#print((cv2.imread(f"images/sample_video_img_000_0.00.jpg"))[0][0][0])




inPath = "images"
outPath = "filtered_images"


for imagePath in os.listdir(inPath):
    #print("imagePath", pixelate(cv2.imread(f"images/{imagePath}")))
    mod_image = pixelate(cv2.imread(f"images/{imagePath}"))

    #os.path.join(outPath, mod_image)
    # filesplit = os.path.split(imagePath)
    # file_name = os.path.splitext(filesplit[1])[0]
    # file_ext = os.path.splitext(imagePath)[1].lower()

    # file_name = file_name + "_modified" + file_ext

    # path = os.path.join(filesplit[0], file_name)

    # cv2.imwrite(path ,src)
    
    cv2.imwrite(os.path.join(outPath, str(imagePath[17]) + str(imagePath[18]) + str(imagePath[19]) + '.jpg'), pixelate(cv2.imread(f"images/{imagePath}")))
    #cv2.imwrite(os.path.join)
    #**save_image(pixelate(cv2.imread(f"images/{imagePath}")), outPath, "modified_image.jpg")
    # with open(os.path.join(inPath, imagePath)) as f:
    #     print(pixelate(f.read()))
#         #print(cv2.imread(f"images/{imagePath}"))
#         for imagePath in cv2.imread(f"images/{imagePath}"):
#             print(imagePath[0][0][0])
            #print(pixelate(imagePath))
            #pixelate(cv2.imread(imagePath))
#             #print("imagePath", imagePath)
            # image = pixelate(cv2.imread(imagePath))
            # filename = 0
            # save_image(image, outPath, "file" +filename)
            # filename+=1
            
            #if cv2.imread(imagePath) == str:
                #print("yay")
            #else:
                #print("what")
            #print("imagePath", imagePath)
            #go through every nine pixel boxes 

# def get_file_key(filename):
#     # Remove all non-digits from the filename
#     key=str(filename[26])+str(filename[27])+str(filename[28])
#     #key = re.sub("[^0-9]", "", filename)
#     return int(key)

# def sort_filenames(all_files):
#     filenames_sorted = []
#     original_filenames = {}
#     for full_filename in all_files:
#         filename, file_extension = os.path.splitext(full_filename)

#         # Save all the files names to be sorted
#         filenames_sorted.append(filename)
#         # Save original full filename in a dictionary for later retrieval
#         original_filenames[filename] = full_filename

#     # Sort the list using our own key
#     filenames_sorted.sort(key=get_file_key)
#     filenames = []
#     for key in filenames_sorted:
#         filenames.append(original_filenames[key])

#     #print(filenames)
#     return filenames

# #sort_filenames("filtered_images")





#print(sorted(os.listdir(mod_images)))

# images = [mod_images_list]
# #print(images)
# frame = cv2.imread(os.path.join(mod_images, images[0]))
# size = images[0].shape[1], images[0].shape[0]

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video = cv2.VideoWriter(video_name, fourcc, 24, size)

# for image in range(len(images)):
#     video.write(images[image])
#     print("x")

cv2.destroyAllWindows()
video.release()




   



    


