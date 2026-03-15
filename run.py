# python run.py --input path/to/blur/images --output path/to/save --weights weights/MLWNet-RealBlur_J.pth


import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
import argparse
from models.MLWNet import build_net

def load_checkpoint(model, weights_path):
    # 使用 map_location='cpu' 避免 GPU 記憶體激增
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # 自動判斷是官方 BasicSR 格式還是自定義格式
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    # 處理 DataParallel/DDP 產生的 'module.' 前綴
    clean_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        clean_state_dict[name] = v
        
    # 載入處理後的參數
    model.load_state_dict(clean_state_dict, strict=True)
    return model

def inference_image(model, img_path, save_path, device):
    # 1. 讀取圖片
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Cannot read image {img_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    img_tensor = img_tensor.to(device)

    # 2. 自動 Padding 處理 (MLWNet 需要長寬是 64 的倍數)
    factor = 64
    h, w = img_tensor.shape[2], img_tensor.shape[3]
    H, W = ((h + factor - 1) // factor) * factor, ((w + factor - 1) // factor) * factor
    pad_h = H - h
    pad_w = W - w
    
    # 使用 reflection padding 避免邊緣偽影
    img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), 'reflect')

    model.eval()
    with torch.no_grad():
        # 3. 模型推論
        # MLWNet 回傳的是 tuple: (x1+inp, x2, x3, x4)
        pred = model(img_tensor)
        if isinstance(pred, (list, tuple)):
            output = pred[0] # 取出解析度最高、已經加上殘差的最終輸出
        else:
            output = pred

    # 4. Un-padding (裁切回原始尺寸)
    output = output[:, :, :h, :w]

    # 5. 後處理 (Clamp -> Numpy -> Save)
    output = torch.clamp(output, 0, 1)
    output = output.cpu().detach().permute(0, 2, 3, 1).numpy().squeeze(0)  # [H, W, 3]
    output = (output * 255.0).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, output)
    print(f"Saved: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='MLWNet Inference Demo')
    parser.add_argument('--input', type=str, default='./input', help='Path to input image or folder')
    parser.add_argument('--output', type=str, default='./results', help='Path to save results')
    parser.add_argument('--weights', type=str, required=True, help='Path to .pkl or .pth file')
    parser.add_argument('--model_name', type=str, default='MLWNet', 
                        choices=['MLWNet'],
                        help='Model name defined in build_net')
    
    args = parser.parse_args()

    # 設定 Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 建立模型
    print(f"Loading model: {args.model_name}...")
    model = build_net(args.model_name) 
    model = model.to(device)

    # 載入權重
    print(f"Loading weights from {args.weights}...")
    model = load_checkpoint(model, args.weights)

    # 準備輸出資料夾
    os.makedirs(args.output, exist_ok=True)

    # 判斷輸入是單張圖片還是資料夾
    if os.path.isfile(args.input):
        img_name = os.path.basename(args.input)
        save_path = os.path.join(args.output, img_name)
        inference_image(model, args.input, save_path, device)
    elif os.path.isdir(args.input):
        valid_exts = {'.jpg', '.png', '.jpeg', '.bmp', '.tif'}
        for img_name in sorted(os.listdir(args.input)):
            ext = os.path.splitext(img_name)[1].lower()
            if ext in valid_exts:
                img_path = os.path.join(args.input, img_name)
                save_path = os.path.join(args.output, img_name)
                inference_image(model, img_path, save_path, device)
    else:
        print("Error: Input path not found.")

if __name__ == '__main__':
    main()