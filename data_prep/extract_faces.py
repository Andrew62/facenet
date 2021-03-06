"""
CLI to walk and extract faces from a top
level directory while maintaining directory structure
"""

import os
import dlib
import argparse
import numpy as np
from PIL import Image
from multiprocessing import Process
import queue


class FaceFinder(object):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def get_faces(self, image_fp, n=1):
        with Image.open(image_fp) as image:
            image_array = np.array(image)
            detections = self.detector(image_array, n)
            for rectangle in detections:
                box = [rectangle.left(), rectangle.top(), rectangle.right(), rectangle.bottom()]
                yield image.crop(box)

    @staticmethod
    def get_fname_no_ext(fp):
        return os.path.splitext(os.path.basename(fp))[0]

    def write_face_chips(self, image_fp, out_dir, **kwargs):
        n = kwargs.pop("n", 1)
        base_name = self.get_fname_no_ext(image_fp)
        for idx, chip in enumerate(self.get_faces(image_fp, n=n)):
            out_fp = os.path.join(out_dir, "{0}_{1}.jpg".format(base_name, idx + 1))
            chip.save(out_fp)


class MySweetMPFaceExtractorThing(Process):
    def __init__(self, name, q):
        super().__init__(name=name)
        self.q = q
        self.ff = FaceFinder()

    def run(self):
        print("{0} starting".format(self.name))
        while not self.q.empty():
            try:
                image_fp, out_dir = self.q.get(timeout=5)
                print(image_fp)
                self.ff.write_face_chips(image_fp, out_dir)
            except queue.Empty:
                return
        print("{0} stopping".format(self.name))




def main():
    parser = argparse.ArgumentParser(description="Extract faces from a directory of images" +
                                     "using DLIB's face detector")
    parser.add_argument("-i", "--in_dir", help="directory to walk for images", required=True,
                        type=str)
    parser.add_argument("-o", '--out_dir', help="Location to save output face chips", type=str,
                        required=True)
    parser.add_argument("-e", "--exts", help="extensions to look for", nargs="+",
                        default=[".jpeg", ".jpg", ".png", ".gif"])
    parser.add_argument("-w", "--n_workers", help="number of processing threads to run",
                        type=int, default=4)
    args = parser.parse_args()
    
    in_dir = args.in_dir
    out_dir = args.out_dir
    
    # make sure the in and out directories end with a slash so we can do an easy replace
    if not out_dir.endswith(os.path.sep):
        out_dir += os.path.sep
    if not in_dir.endswith(os.path.sep):
        in_dir += os.path.sep

    q = queue.Queue()
    for d, _, files in os.walk(in_dir):
        out_current_dir = d.replace(in_dir, out_dir)
        os.makedirs(out_current_dir, exist_ok=True)
        for f in files:
            if any(map(lambda x: f.endswith(x), args.exts)):
                image_fp = os.path.join(d, f)
                q.put((image_fp, out_current_dir))

    processes = []
    for i in range(args.n_workers):
        p = MySweetMPFaceExtractorThing(str(i + 1), q)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    print("Done")


if __name__ == "__main__":
    main()

