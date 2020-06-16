import os
import cv2

# load all tagged images
path = os.getcwd()
imgs_path = os.path.join(path, "images", "orig", "tagged_training")
bib_path = os.path.join(images_path, "bibs")
# try:
#     os.mkdir(bib_path)
# except OSError as error:
#     print(error)

target_class = '1'  # 0: face, 1: bib

def crop_bib_images(images_path):
    seq = 1
    with os.scandir(images_path) as it:
        for entry in it:
            if entry.name.endswith('JPG'):
                imgpath = os.path.splitext(entry.path)[0]
                image = cv2.imread(entry.path)
                (H, W) = image.shape[:2]
                print(entry.name)
                with open(imgpath+".txt", 'r') as f:
                    boxes = f.readlines()
                    boxes_pos = [box for box in boxes if box.split(' ', 1)[0] == target_class]
                    for pos in boxes_pos:
                        (cx, cy, w, h) = list(map(float, pos.split(' ')[1:]))
                        x1 = int((cx - (w / 2)) * W)
                        y1 = int((cy - (h / 2)) * H)
                        x2 = int((cx + (w / 2)) * W)
                        y2 = int((cy + (h / 2)) * H)
                        cropped = image[y1:y2, x1:x2]
                        fn = os.path.join(bib_path, "{0:05d}.jpg".format(seq))
                        cv2.imwrite(fn, cropped)
                        seq += 1

#crop_bib_images(imgs_path)