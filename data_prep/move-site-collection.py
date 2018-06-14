import os
import csv
import zipfile
import io
import dlib
from PIL import Image
import numpy as np

indir = "../fixtures/faces/site-collection"
out_dir = "../fixtures/faces/"

detector = dlib.get_frontal_face_detector()

latest_archive = sorted([os.path.join(indir, f) for f in os.listdir(indir)], key=lambda x: os.stat(x).st_mtime, reverse=True)[0]


with zipfile.ZipFile(latest_archive) as zfile:
    reader = csv.reader(zfile.read("img-updates.csv").decode().strip().split("\n"))
    for name, img in reader:
        data = zfile.read("media/" +img)
        jpg = Image.open(io.BytesIO(data))
        img_array = np.array(jpg)
        for rect in detector(img_array):
            bbox = [rect.left(), rect.top(), rect.right(), rect.right()]
            face = jpg.crop(bbox)
            if face:
                fname = os.path.basename(img)
                out_person_dir = os.path.join(out_dir, name)
                os.makedirs(out_person_dir, exist_ok=True)
                outf = os.path.join(out_person_dir, fname)
                face.save(outf)
print("done")
