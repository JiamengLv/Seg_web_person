import os
import base64
import numpy as np
from astropy.io import fits
from PIL import Image
from io import BytesIO

def read_fits(path):
    hdul = fits.open(path)
    image_data = hdul[0].data
    hdul.close()
    return image_data

def remove_white_border(img, white_value=255):

    mask = (img == white_value)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]

def crop_img(image_paths, patch ,output_dir):
    for image_path in image_paths:
        image_base = os.path.basename(image_path)
        file_name, file_ext = os.path.splitext(image_base)
        data = read_tif_data(image_path)
        vmin, vmax = data.min(), data.max()
        img_data = np.clip(data, vmin, vmax)
        image_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
        image_data = (image_data * 255).astype(np.uint8)
        # image_data = remove_white_border(image_data)

        H, W = image_data.shape
        # nH, nW = H // patch, W // patch
        nH = int(np.ceil(H / patch))
        nW = int(np.ceil(W / patch))
        for i in range(0, (nH+1)):
            for j in range(0, (nW+1)):
                xs = j*patch
                ys = i*patch
                xe = (j+1)*patch
                ye = (i+1)*patch
                if xe >= W:
                    xe = W
                    xs = xe - patch
                if ye >= H:
                    ye = H
                    ys = ye - patch
                crop_patch = image_data[ys:ye, xs:xe]
                if crop_patch.shape[0] == patch and crop_patch.shape[1] == patch:
                    hduout = fits.PrimaryHDU(crop_patch)
                    hduout.writeto(os.path.join(output_dir, file_name+f'_{j}_{i}.fits'), overwrite=True)


def detection_img(image_paths):
    image_ids = os.listdir(image_paths)
    return len(image_ids)


def fits_to_jpg(fits_path):
    hdul = fits.open(fits_path)
    image_data = hdul[0].data
    hdul.close()

    image = Image.fromarray(image_data)
    image = image.convert('RGB')

    buf = BytesIO()
    image.save(buf, format='JPEG', quality=90)
    b = buf.getvalue()
    return base64.b64encode(b).decode('utf-8')



