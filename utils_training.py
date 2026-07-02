def get_mask_path(image_id, trainset):
    """
    优先返回 Corrections 下的 mask 路径，否则返回 Annotations 下的。
    """
    corrections_path = os.path.join(trainset, 'Corrections', image_id)
    annotations_path = os.path.join(trainset, 'Annotations', image_id)
    if os.path.exists(corrections_path):
        return corrections_path
    return annotations_path
import os
import shutil
import base64
import numpy as np
from astropy.io import fits
from fastapi import HTTPException
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import UNet
from PIL import Image
from io import BytesIO
import cv2

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

def read_fits(p):
    hdul = fits.open(p)
    image_data = hdul[0].data
    hdul.close()
    return image_data

def read_labels(p):
    labels = []
    with open(p, 'w') as fr:
        lines = fr.readlines()
        for line in lines:
            cx, cy, rii = line.split(' ')
            labels.append([cx, cy, rii])
    return labels


def generate_mask(image_id, coords, mask, out_path, pad=16):
    h, w = mask.shape
    padding_mask = np.zeros((h+2*pad, w+2*pad),dtype=np.uint8)
    padding_mask[pad:pad + h, pad:pad + w] = mask
    for coord in coords:
        xc, yc, r = int(float(coord[0])), int(float(coord[1])), int(float(coord[2]))
        if r <= 0:
            continue
        if xc < 0 or yc < 0 or xc >= w or yc >= h:
            continue
        xc_pad = xc+pad
        yc_pad = yc+pad
        # print(f'{xc_pad}--{yc_pad}')

        x0 = xc_pad-r
        y0 = yc_pad-r
        x1 = xc_pad+r
        y1 = yc_pad+r


        yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
        circle = (xx - xc_pad) ** 2 + (yy - yc_pad) ** 2 <= (r ** 2)
        padding_mask[y0:y1 + 1, x0:x1 + 1] |= circle.astype(np.uint8)

    final_mask = padding_mask[pad:pad + h, pad:pad+w]
    return final_mask

    # hduout = fits.PrimaryHDU(final_mask)
    # hduout.writeto(os.path.join(out_path, image_id+'.fits'), overwrite=True)


def train_dataset(d, m, fp, tp, patch):
    for image_id, coords in d.items():
        try:
            mask = m[image_id]
        except Exception:
            mask = np.zeros((patch, patch), dtype=np.uint8)

        final_mask = generate_mask(image_id, coords, mask, os.path.join(tp, 'Annotations'))
        if np.sum(final_mask) == 0:
            print(f"[delete] {image_id} (mask=0)")
            continue

        mask_path = os.path.join(tp, 'Annotations', image_id + '.fits')
        fits.PrimaryHDU(final_mask).writeto(mask_path, overwrite=True)
        shutil.copy(
            os.path.join(fp, 'images', image_id+'.fits'),
            os.path.join(tp, 'Images', image_id+'.fits')
        )

def remove_white_border(img, white_value=255):
    mask = (img != white_value)
    coords = np.argwhere(mask)
    if coords.size == 0:
        # return img
        h, w = img.shape
        return 0, h, 0, w
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    # return img[y0:y1, x0:x1]
    return y0, y1, x0, x1

def dataloader(trainset):
    images = []
    masks = []
    image_ids = os.listdir(os.path.join(trainset, 'Images'))

    for image_id in image_ids:
        image_data = read_fits(os.path.join(trainset, 'Images', image_id))
        mask_data = read_fits(os.path.join(trainset, 'Annotations', image_id))

        images.append(image_data)
        masks.append(mask_data)

    return images, masks


def prepare_data(trainset, target_size=256):


    class FITSDataset(Dataset):
        def __init__(self, root_dir, target_size):
            self.root_dir = root_dir
            self.image_dir = os.path.join(root_dir, 'Images')
            self.image_ids = sorted(os.listdir(self.image_dir))
            self.target_size = target_size

        def __len__(self):
            return len(self.image_ids)

        def __getitem__(self, idx):
            image_id = self.image_ids[idx]
            image_path = os.path.join(self.image_dir, image_id)
            mask_path = get_mask_path(image_id, self.root_dir)

            image_data = read_fits(image_path).astype(np.float32)
            mask_data = read_fits(mask_path)

            y0, y1, x0, x1 = remove_white_border(image_data, white_value=255)
            image_crop = image_data[y0:y1, x0:x1]
            mask_crop = mask_data[y0:y1, x0:x1]

            image_norm = image_crop / 255.0
            image_resized = cv2.resize(image_norm, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask_crop, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

            mask_bin = (mask_resized > 0).astype(np.uint8)

            image_tensor = torch.from_numpy(image_resized).float().unsqueeze(0)  # [1, H, W]
            mask_tensor = torch.from_numpy(mask_bin).long().unsqueeze(0)  # long类型，适配CrossEntropyLoss

            return image_tensor, mask_tensor

    dataset = FITSDataset(trainset, target_size)
    all_mask_vals = []
    for i in range(len(dataset)):
        _, mask = dataset[i]
        all_mask_vals.extend(mask.numpy().flatten().tolist())
    all_mask_vals = np.array(all_mask_vals)
    print('[SUMMARY] 全部训练mask像素唯一值:', np.unique(all_mask_vals, return_counts=True))
    training_dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    return training_dataloader

def load_model(model, path):
    path=path+'/unet_model.pth'
    model.load_state_dict(torch.load(path))
    model.eval()  # 切换到评估模式
    print(f"模型权重已加载自 {path}")

def training_unet(model, model_path, training_dataloader, epochs = 10, lr = 1e-3, cb=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    import tqdm

    model.train()


    # 新增：保存loss到文件
    loss_log_path = os.path.join(model_path, 'unet_train_loss.txt')
    with open(loss_log_path, 'w') as flog:
        for epoch in tqdm.tqdm(range(epochs), desc="Training Epochs"):
            running_loss = 0.0
            for inputs, labels in tqdm.tqdm(training_dataloader, desc="Training Batches"):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                # labels 已经是 long 类型且为0/1
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)

                loss = criterion(outputs, labels.squeeze(1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / max(1, len(training_dataloader))
            if cb:
                pct = int((epoch + 1) / epochs * 100)
                cb(pct, f'epoch{epoch+1}/{epochs}, loss={avg_loss:.4f}')

            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
            flog.write(f"{epoch+1}\t{avg_loss:.6f}\n")

    save_name = f"unet_model.pth"
    save_path = os.path.join(model_path, save_name)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    print(f"Saving model to {save_path}...")
    torch.save(model.state_dict(), save_path)

    # 可视化训练损失曲线
    try:
        import matplotlib.pyplot as plt
        epochs_list = []
        loss_list = []
        with open(loss_log_path, 'r') as flog:
            for line in flog:
                epoch_num, loss_val = line.strip().split('\t')
                epochs_list.append(int(epoch_num))
                loss_list.append(float(loss_val))

        plt.figure()
        plt.plot(epochs_list, loss_list, marker='o')
        plt.title('UNet Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.savefig(os.path.join(model_path, 'unet_training_loss_curve.png'))
        plt.close()
        # keshihua jieguo 

        plt.subplot(1, 3, 1)
        plt.imshow(inputs[0, 0].cpu().numpy(), cmap='gray')
        plt.title('Input Image')
        plt.subplot(1, 3, 2)
        plt.imshow(labels[0, 0].cpu().numpy(), cmap='gray')
        plt.title('Ground Truth Mask')
        plt.subplot(1, 3, 3)
        pred_mask = torch.argmax(outputs, dim=1)[0].cpu().numpy()
        plt.imshow(pred_mask, cmap='gray')
        plt.title('Predicted Mask')
        plt.savefig(os.path.join(model_path, 'unet_training_sample.png'))
        plt.close()


    except Exception as e:
        print(f"可视化训练损失曲线失败: {e}")


def start_training_impl(payload, progress_cb, p):
    training_dataset_path = os.path.abspath(payload['training_data_path'])
    training_dataloader = prepare_data(training_dataset_path)
    model = UNet(in_channels=1, out_channels=2)
    # 微调：如果已有模型权重则加载
    model_path = os.path.join(p, 'unet_model.pth')
    if os.path.exists(model_path):
        print(f"[INFO] 加载已有模型权重进行微调: {model_path}")
        model.load_state_dict(torch.load(model_path))
    training_unet(model, p, training_dataloader, cb=progress_cb)

def run_detection(fits_path, model, sp, id):
    if model:
        idata = read_fits(fits_path)
        idata = idata.astype(np.float32)
        image_data = idata / 255.0
        image_data = image_data.astype(np.float32)
        img_ipt = torch.from_numpy(image_data)
        img_ipt = img_ipt.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            pred = model(img_ipt)
            pred = torch.argmax(pred, dim=1)[0].cpu().numpy()
            # pred = model(img_ipt)[0, 0].cpu().numpy()
        binary = pred.astype(np.uint8)
        hduout = fits.PrimaryHDU(binary)
        hduout.writeto(os.path.join(sp, id+'.fits'), overwrite=True)

        import matplotlib.pyplot as plt
        
        plt.subplot(1, 3, 1)
        plt.imshow(img_ipt[0, 0].cpu().numpy(), cmap='gray')
        plt.title('Input Image')
        plt.subplot(1, 3, 2)
        plt.imshow(pred[0, 0].cpu().numpy(), cmap='gray')
        plt.title(' Mask')

        plt.savefig(os.path.join('unet_training_sample.png'))
        plt.close()

        return binary

    else:
        return None

def detection_img_full(image_data, model):
    image_data = image_data.astype(np.float32)
    img_ipt = torch.from_numpy(image_data)
    img_ipt = img_ipt.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        pred = model(img_ipt)
        pred = torch.argmax(pred, dim=1)[0].cpu().numpy()
    print('[DEBUG] Infer pred unique:', np.unique(pred))
    binary = pred.astype(np.uint8)
    return binary



def mask_to_b64(msk):
    m = (msk > 0).astype(np.uint8)*255
    im = Image.fromarray(m, mode='L')
    buf = BytesIO()
    im.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def nparray_to_b64(arr):
    arr_norm = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8) * 255
    im = Image.fromarray(arr_norm.astype(np.uint8), mode='L')
    buf = BytesIO()
    im.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def load_validation_model(p):
    if p:
        model = UNet(in_channels=1, out_channels=2)
        load_model(model, p)
        return model
    else:
        return None
