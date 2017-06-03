import numpy as np
from PIL import Image
import os.path as osp


def get_thumbnail(fp, size=(100, 100)):
    with Image.open(fp) as image:
        return np.asarray(image.resize(size))


def make_sprite(image_fps, out_image, thumbnail_size=(100, 100)):
    """
    Make a sprite image for tensorboard
    """
    images = []
    for image_fp in image_fps:
        images.append(get_thumbnail(image_fp, thumbnail_size))

    sprite_size = int(np.ceil(np.sqrt(len(images))))

    # make sure the jpeg is square
    while len(images) < (sprite_size**2):
        blank_image = np.zeros((thumbnail_size[0], thumbnail_size[1], 3), dtype=np.uint8)
        images.append(blank_image)

    rows = []
    for idx in range(sprite_size):
        start = idx * sprite_size
        end = start + sprite_size
        row = images[start: end]
        rows.append(np.concatenate(row, 1))
    sprite = Image.fromarray(np.concatenate(rows, 0))
    sprite.save(out_image)


def sprite_metadata(file_paths, out_f):
    """
    Write the metadata file for tensorboard
    """
    with open(out_f, "w") as target:
        for fp in file_paths:
            name = osp.basename(osp.dirname(fp))
            target.write("{}\n".format(name))
