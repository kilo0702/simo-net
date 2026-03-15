import os
import torch
import argparse
from torch.backends import cudnn
from models.MLWNet import build_net  # 注意：請確保對應你新建立/更改的 MLWNet.py
from train import _train
from eval import _eval

def main(args):
    cudnn.benchmark = True

    # 根據 model_name 自動設定 wavelet loss（若使用者未明確指定則按 model 決定）
    if args.model_name == 'MLWNet-B':
        args.use_wavelet_loss = True
    elif args.model_name == 'MLWNet-S':
        args.use_wavelet_loss = False

    if not os.path.exists('results/'):
        os.makedirs('results/')
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net(args.model_name)
    if torch.cuda.is_available():
        model.cuda()
        
    if args.mode == 'train':
        _train(model, args)
    elif args.mode == 'test':
        _eval(model, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_wavelet_loss', type=lambda x: x.lower() == 'true', default=True,
                        help='Use wavelet reconstruction loss (default: True for MLWNet-B)')
    
    # [修改] 替換為 B 和 S 版本
    parser.add_argument('--model_name', default='MLWNet-B', choices=['MLWNet-B', 'MLWNet-S'], type=str)
    parser.add_argument('--data_dir', type=str, default='dataset/GOPRO')
    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)

    # Train
    parser.add_argument('--batch_size', type=int, default=8) # 建議配合作者預設的 8
    parser.add_argument('--learning_rate', type=float, default=9e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_epoch', type=int, default=3000)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=50) # 配合你驗證的頻率，改為 50
    parser.add_argument('--valid_freq', type=int, default=50)
    parser.add_argument('--resume', type=str, default='')
    
    # 已將無用的 gamma 與 lr_steps 刪除

    # Test
    parser.add_argument('--test_model', type=str, default='weights/MLWNet-B.pkl')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', args.model_name, 'weights/')
    args.result_dir = os.path.join('results/', args.model_name, 'result_image/')
    print(args)
    main(args)