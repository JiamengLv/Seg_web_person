import os
import requests
import numpy as np
from astropy.io import fits
import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources, detect_threshold
from photutils.utils import circular_footprint
from skimage.util import view_as_blocks

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources
from photutils.utils import circular_footprint
from scipy.ndimage import binary_dilation

def make_source_mask(data, nsigma=3, npixels=5, dilate_size=11):
    """
    自定义实现 make_source_mask（兼容 photutils >= 1.5）
    """
    # 1. 计算背景统计
    _, median, std = sigma_clipped_stats(data, sigma=3.0)
    
    # 2. 计算检测阈值
    threshold = median + (nsigma * std)
    
    # 3. 检测源（生成 segmentation image）
    segm = detect_sources(data, threshold, npixels=npixels)
    
    if segm is None:
        return np.zeros(data.shape, dtype=bool)
    
    # 4. 获取布尔掩膜（有源区域为 True）
    mask = segm.data > 0
    
    # 5. 膨胀掩膜（使用 scipy）
    if dilate_size > 0:
        footprint = circular_footprint(radius=dilate_size // 2)
        mask = binary_dilation(mask, structure=footprint)
    
    return mask

# 下载 DESI g 波段 FITS 文件
def download_fits(url, save_path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

# 生成中心天体的 mask
def generate_mask(image_data):
    mask = make_source_mask(image_data, nsigma=2, npixels=5, dilate_size=11)
    return mask.astype(np.uint8)

# 裁剪为 512x512 patch 并保存
def save_patches(image, mask, out_dir, prefix):
    os.makedirs(os.path.join(out_dir, 'Images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'Annotations'), exist_ok=True)
    H, W = image.shape
    patch = 256
    nH = H // patch
    nW = W // patch
    idx = 0
    for i in range(nH):
        for j in range(nW):
            img_patch = image[i*patch:(i+1)*patch, j*patch:(j+1)*patch]
            mask_patch = mask[i*patch:(i+1)*patch, j*patch:(j+1)*patch]
            if img_patch.shape == (patch, patch) and mask_patch.shape == (patch, patch):
                img_name = f'{prefix}_{i}_{j}.fits'
                fits.PrimaryHDU(img_patch).writeto(os.path.join(out_dir, 'Images', img_name), overwrite=True)
                fits.PrimaryHDU(mask_patch).writeto(os.path.join(out_dir, 'Annotations', img_name), overwrite=True)
                idx += 1
    print(f'Saved {idx} patches.')

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# 可视化一个 image 和 mask patch

def show_example(image_dir, mask_dir, idx=0):
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    if not image_files or not mask_files:
        print('No files found!')
        return
    img_path = os.path.join(image_dir, image_files[idx])
    mask_path = os.path.join(mask_dir, mask_files[idx])
    img = fits.getdata(img_path)
    mask = fits.getdata(mask_path)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Image')
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title('Mask')
    axs[2].imshow(img, cmap='gray')
    axs[2].imshow(mask, cmap='Reds', alpha=0.4)
    axs[2].set_title('Overlay')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('example_visualization.png')


if __name__ == '__main__':
    url = 'https://casdc.china-vo.org/mirror/DESI-DR10/south/coadd/007/0070m115/legacysurvey-0070m115-image-g.fits.fz'
    fits_path = 'legacysurvey-0070m115-image-g.fits.fz'
    out_dir = r'D:\Program\月球大模型-地球化学所\标签修改\training_data_path'

    if not os.path.exists(fits_path):
        print('Downloading FITS file...')
        download_fits(url, fits_path)
    else:
        print('FITS file already exists.')

    with fits.open(fits_path) as hdul:
        image_data = hdul[1].data.astype(np.float32)

    mask = generate_mask(image_data)

    save_patches(image_data, mask, out_dir, 'desi_g')
    print('Done.')

    image_dir = os.path.join(out_dir, 'Images')
    mask_dir = os.path.join(out_dir, 'Annotations')

    show_example(image_dir, mask_dir, idx=0)