import torch
from torch.utils.data import DataLoader, Dataset
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm


def rgb_to_y_float(img_float):
    """
    將 float32 RGB 圖片轉換為 YCbCr Y 通道（float32）
    在 float 空間計算，避免 uint8 量化誤差。

    Args:
        img_float: numpy array, shape (H, W, 3), float32, range [0, 1]

    Returns:
        Y channel: numpy array, shape (H, W), float32
                   range ≈ [16/255, 235/255]，即 [0.0627, 0.9216]
    """
    # ITU-R BT.601，與原本 uint8 版本係數一致，但在 [0,1] 空間操作
    R = img_float[:, :, 0]
    G = img_float[:, :, 1]
    B = img_float[:, :, 2]
    Y = (64.738 * R + 129.057 * G + 25.064 * B) / 255.0 + 16.0 / 255.0
    # clip 到合法範圍
    Y = np.clip(Y, 16.0 / 255.0, 235.0 / 255.0)
    return Y  # float32，range [0.0627, 0.9216]


class ValDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir):
        self.blur_dir = blur_dir
        self.sharp_dir = sharp_dir

        self.image_files = sorted([
            f for f in os.listdir(blur_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        blur_img  = Image.open(os.path.join(self.blur_dir,  file_name)).convert('RGB')
        sharp_img = Image.open(os.path.join(self.sharp_dir, file_name)).convert('RGB')
        return TF.to_tensor(blur_img), TF.to_tensor(sharp_img)


def _valid(model, args, epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 明確組出 blur / sharp 路徑，不依賴 string replace
    test_root = args.data_dir
    blur_dir  = os.path.join(test_root, 'test', 'blur')
    sharp_dir = os.path.join(test_root, 'test', 'sharp')

    if not os.path.exists(blur_dir):
        # fallback：有些資料集直接放在 data_dir 下
        blur_dir  = os.path.join(test_root, 'blur')
        sharp_dir = os.path.join(test_root, 'sharp')

    if not os.path.exists(blur_dir):
        raise FileNotFoundError(
            f"[_valid] Cannot find blur directory. Tried:\n"
            f"  {os.path.join(test_root, 'test', 'blur')}\n"
            f"  {blur_dir}"
        )

    val_dataset = ValDataset(blur_dir, sharp_dir)
    val_loader  = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_worker, pin_memory=True
    )

    model.eval()
    psnr_adder = Adder()
    ssim_adder = Adder()

    print(f'\nValidation on {len(val_dataset)} images')
    print(f'  blur : {blur_dir}')
    print(f'  sharp: {sharp_dir}')
    print('Computing PSNR/SSIM on YCbCr Y channel (float32, data_range=1.0)')

    progress_bar = tqdm(val_loader, desc=f"Valid Epoch {epoch}", ncols=110)

    with torch.no_grad():
        for idx, data in enumerate(progress_bar):
            input_img, label_img = data
            input_img = input_img.to(device)

            # Padding to multiple of 64
            factor = 64
            h, w = input_img.shape[2], input_img.shape[3]
            H = ((h + factor - 1) // factor) * factor
            W = ((w + factor - 1) // factor) * factor
            input_padded = F.pad(input_img, (0, W - w, 0, H - h), 'reflect')

            # Inference
            pred = model(input_padded)
            output_padded = pred[0] if isinstance(pred, (list, tuple)) else pred

            # Un-padding → clamp → numpy float32 [0,1]
            pred_np  = torch.clamp(output_padded[:, :, :h, :w], 0, 1) \
                           .cpu().numpy().squeeze(0).transpose(1, 2, 0)
            label_np = label_img.numpy().squeeze(0).transpose(1, 2, 0)

            # ★ 在 float32 空間轉 Y channel，避免兩次 uint8 量化誤差
            pred_y  = rgb_to_y_float(pred_np)
            label_y = rgb_to_y_float(label_np)

            # ★ data_range=1.0：Y 的值域已是 [0,1] 的子集，用 1.0 才正確
            psnr_val = peak_signal_noise_ratio(label_y, pred_y, data_range=1.0)
            ssim_val = structural_similarity(
                label_y, pred_y, data_range=1.0, win_size=11
            )

            psnr_adder(psnr_val)
            ssim_adder(ssim_val)

            progress_bar.set_postfix(
                Avg_PSNR=f"{psnr_adder.average():.2f}",
                Avg_SSIM=f"{ssim_adder.average():.4f}"
            )

    return psnr_adder.average(), ssim_adder.average()