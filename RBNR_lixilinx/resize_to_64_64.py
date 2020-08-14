import os
import fnmatch
from PIL import Image

def resize(in_path, out_path, size=64):
    img_dim = (size, size)
    # i = 0
    for f in os.scandir(in_path):
        if fnmatch.fnmatch(f, '*.jpg'):
            with Image.open(f.path) as img:
                img = img.resize(img_dim, resample=1, reducing_gap=3)
                img.save(os.path.join(out_path, f.name))
                print(os.path.join(out_path, f.name))

in_path = os.path.join(os.getcwd(), 'images')
out_path = os.path.join(in_path, '64')
resize(in_path, out_path)