import os
import torch
from tqdm import tqdm
 
from data.data_load import train_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter

# [修改] 改用針對 RealBlur 優化過的驗證腳本
from valid_realblur import _valid 

import torch.nn.functional as F


def _train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from losses import SRN_loss
    criterion_pixel = SRN_loss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay,  # 建議從 args 傳入 1e-4 或 0
                                  betas=(0.9, 0.999))

    # 載入資料
    print(f"Loading Image data from: {args.data_dir}")
    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    
    max_iter = len(dataloader)
    total_iter = args.num_epoch * max_iter  # 正確：以 iter 為單位
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iter, eta_min=1e-7)
    epoch = 1
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        model.load_state_dict(state['model'])
        print('Resume from %d'%epoch)
        epoch += 1

    writer = SummaryWriter()
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    epoch_wav_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    iter_wav_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr=-1

    for epoch_idx in range(epoch, args.num_epoch + 1):
        
        model.train()
        epoch_timer.tic()
        iter_timer.tic()
        
        # [優化] tqdm 加上 ncols 讓顯示更整齊
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_idx}/{args.num_epoch}", ncols=120)
        
        for iter_idx, batch_data in enumerate(progress_bar):

            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()
            # MLWNet回傳 (x1+inp, x2, x3, x4)
            pred_img = model(input_img)

            # ==========================================
            # Multi-scale Content Loss (SRN_loss)
            # ==========================================
            loss_content = criterion_pixel(pred_img, label_img)

            # ==========================================
            # Multi-scale FFT Loss
            # ==========================================
            # 原始 MLWNet-GoPro 設定檔沒有使用 FFT Loss
            loss_fft = torch.tensor(0.0).to(device)

            # ==========================================
            # Wavelet Reconstruction Loss
            # ==========================================
            if args.use_wavelet_loss:
                loss_wavelet = model.module.get_wavelet_loss() if hasattr(model, 'module') else model.get_wavelet_loss()
            else:
                loss_wavelet = torch.tensor(0.0).to(device)
            # 總 Loss
            loss = loss_content + loss_wavelet

            loss.backward()
            optimizer.step()
            scheduler.step()  # 每個 iter 更新，而非每個 epoch


            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())
            iter_wav_adder(loss_wavelet.item())

            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())
            epoch_wav_adder(loss_wavelet.item())

            if (iter_idx + 1) % args.print_freq == 0:
                lr = check_lr(optimizer)
                # [修改] 統一 tqdm 的標籤顯示名稱
                progress_bar.set_postfix(
                    LR=f"{lr:.8f}",
                    Loss_Content=f"{iter_pixel_adder.average():.4f}",
                    Loss_FFT=f"{iter_fft_adder.average():.4f}",
                    Loss_Wav=f"{iter_wav_adder.average():.4f}"
                )

                writer.add_scalar('Train/Loss_Content', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                writer.add_scalar('Train/Loss_FFT',     iter_fft_adder.average(),   iter_idx + (epoch_idx-1)* max_iter)
                writer.add_scalar('Train/Loss_Wavelet', iter_wav_adder.average(),   iter_idx + (epoch_idx-1)* max_iter)

                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()
                iter_wav_adder.reset()
                
        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        if epoch_idx % args.save_freq == 0 or epoch_idx == 1:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, save_name)
                        
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f Epoch Wavelet Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average(), epoch_wav_adder.average()))

        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        epoch_wav_adder.reset()
        # scheduler.step() 已移至 iter 層級

        # [修改] 強制設定驗證頻率為第 1 個 epoch，以及後續每 50 個 epoch (50, 100, 150...)
        if epoch_idx == 1 or epoch_idx % 50 == 0:
            val_psnr, val_ssim = _valid(model, args, epoch_idx)
            print('%03d epoch \n Average RealBlur PSNR %.2f dB, Average RealBlur SSIM %.4f' % (epoch_idx, val_psnr, val_ssim))
            
            # [修改] 統一 TensorBoard 驗證標籤 (修正 GOPRO 為 RealBlur，並加上 Eval/ 前綴)
            writer.add_scalar('Eval/PSNR_RealBlur', val_psnr, epoch_idx)
            writer.add_scalar('Eval/SSIM_RealBlur', val_ssim, epoch_idx)
            
            if val_psnr >= best_psnr:
                best_psnr = val_psnr
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
                print(f"[*] New Best PSNR Model Saved: {best_psnr:.2f} dB")
                
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)