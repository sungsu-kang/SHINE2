from torch.utils.data import DataLoader
from torch import nn
from Utils.Dataloader_mrc import *
from Nets.Blindspot_Net import *
from Utils.patch_generator_5frame import *
from Utils.Utils import *
from tempfile import mkdtemp
from Nets.UNet import UNet
import cv2 as cv
import os
import torch
import pytorch_lightning as pl
import numpy as np
import torchvision  

@torch.jit.script
def shuffle_blocks(image_tensor, kernel: int):
    n, c, h, w = image_tensor.shape
    if kernel == 1:
        new_image = image_tensor
    else:
        new_image = torch.empty_like(image_tensor)
        random_pad_i = torch.randint(0, kernel, (1,)).item()
        random_pad_j = torch.randint(0, kernel, (1,)).item()
        for i in range(0, h + kernel, kernel):
            for j in range(0, w + kernel, kernel):
                h_start = max(i + random_pad_i - kernel, 0)
                w_start = max(j + random_pad_j - kernel, 0)
                h_end = min(i + random_pad_i, h)
                w_end = min(j + random_pad_j, w)

                # Check if the block size is at least 1x1
                if h_end - h_start > 0 and w_end - w_start > 0:
                    block = image_tensor[:, :, h_start:h_end, w_start:w_end]
                    flat_block = block.reshape(n, c, -1)
                    num_pixels = flat_block.shape[2]
                    pixel_indices = torch.randperm(num_pixels)
                    shuffled_block = flat_block[:, :, pixel_indices]
                    reshaped_block = shuffled_block.reshape(n, c, h_end - h_start, w_end - w_start)
                    new_image[:, :, h_start:h_end, w_start:w_end] = reshaped_block

    return new_image.detach()

class L1_Charbonnier_loss(nn.Module):
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6
    
    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.mean(error) 
        return loss

class hfm(torch.nn.Module):
    def __init__(self, scale=2):
        super(hfm, self).__init__()
        self.scale = scale
        self.downscale = nn.AvgPool2d(scale)
        self.upscale = nn.Upsample(scale_factor=scale,mode='bilinear')

    def forward(self,x):
        _, _, nH, nW = x.shape
        if nH % self.scale != 0 or nW % self.scale != 0:
            x = F.pad(x, (0, self.scale - nH % self.scale, self.scale - nW % self.scale,0))
        x =  x-self.upscale(self.downscale(x))
        if nH % self.scale != 0 or nW % self.scale != 0:
            x = x[:, :, 0:nH, 0:nW]
        return x

@torch.jit.script
def gauss_noise_torch(img, noise_scaler: float = 0.25):
    #img = torch_zscore_normalize(img)
    
    # sigma is now a scalar tensor
    sigma = torch.rand(1) * noise_scaler

    # Generate Gaussian noise
    noise = sigma.to(img.device) * torch.randn_like(img)

    # Add noise to the image
    out = img + noise.to(img.device)
    #out = torch.clamp(out, 0, 1)
    out = torch_zscore_normalize(out)
    return out

class TEM_denoiser_main(pl.LightningModule):
    def __init__(self,
                network,
                in_channels,
                out_channels,
                frame_num,
                img_size,
                training_path,
                save_folder,
                time_stamp,
                model_type,
                learning_rate,
                batch_size, 
                lossF,
                beta1,
                beta2,
                eps,
                weight_decay,
                total_epochs,
                trainset,
                validationset,
                testset,
                mean_train,
                std_train,
                maximum_train,
                additional_dilation_i,
                additional_dilation_j,
                second_stage_network=None,
                second_additional_dilation_i=0,
                second_additional_dilation_j=0,
                ):
        super(TEM_denoiser_main, self).__init__()
        self.frame_num=frame_num
        self.prepare_data_per_node=False
        self.save_folder=save_folder
        self.time_stamp=time_stamp
        self.model_type=model_type
        self.learning_rate=learning_rate
        self.loss_F=lossF
        self.beta1=beta1
        self.beta2=beta2
        self.eps=eps
        self.weight_decay=weight_decay
        self.batch_size=batch_size
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.training_path=training_path
        self.img_size=img_size
        self.total_epochs=total_epochs
        self.Validationset=validationset
        self.Trainset=trainset
        self.Testset=testset
        self.loss_function_forward=self.loss_function()
        self.model=network
        self.second_model = second_stage_network
        self.two_stage = second_stage_network is not None
        self.mean_train=mean_train
        self.std_train=std_train
        self.maximum_train=maximum_train
        self.additional_dilation=np.maximum(additional_dilation_i, additional_dilation_j)
        self.second_additional_dilation = (
            np.maximum(second_additional_dilation_i, second_additional_dilation_j)
            if self.two_stage
            else 0
        )
        
    def forward(self, x, shuffle=0):
        out = self.model(x, shuffle=shuffle)
        return out

    def forward_second_stage(self, x, shuffle=0):
        if not self.two_stage:
            raise RuntimeError("forward_second_stage called without second_stage_network")
        return self.second_model(x, shuffle=shuffle)

    def _mrc_tile_denoise_frame(
        self,
        frames,
        inner_net,
        shuffle_int,
        img_size,
        slice_pad,
        gain_value,
    ):
        """
        frames: (1, C, H, W) raw multi-frame window (not yet clip/normalize).
        Returns (1, 1, H, W) on CPU after z-score recovery, same as test_step.
        """
        for i in range(frames.shape[1]):
            frames[:, i, :, :] = torch.from_numpy(
                clip_top_3_percent(frames[:, i, :, :].detach().cpu().numpy())
            ).to(frames.device)
        frames, nonsacle = numpy_zscore_normalize_test(frames)
        denoised = torch.zeros(
            1,
            1,
            frames.shape[2],
            frames.shape[3],
            dtype=torch.float32,
        ).cpu()
        for ii in range(0, frames.size(2), img_size):
            for jj in range(0, frames.size(3), img_size):
                si = max(0, ii - slice_pad)
                ei = min(frames.size(2), ii + img_size + slice_pad)
                sj = max(0, jj - slice_pad)
                ej = min(frames.size(3), jj + img_size + slice_pad)
                xij = frames[:, :, si:ei, sj:ej].type_as(gain_value)
                yij = inner_net(xij, shuffle=shuffle_int).squeeze().detach().cpu()
                top_pad = ii - si
                left_pad = jj - sj
                bottom_pad = ei - (ii + img_size)
                right_pad = ej - (jj + img_size)
                yij = yij[top_pad : yij.shape[-2] - bottom_pad, left_pad : yij.shape[-1] - right_pad]
                denoised[:, :, ii : ii + yij.shape[-2], jj : jj + yij.shape[-1]] = yij
        return numpy_zscore_recover(denoised, nonsacle.cpu())

    def _denoise_full_mrc_movie(
        self,
        img,
        inner_net,
        shuffle_int,
        img_size,
        slice_pad,
        gain_value,
        desc,
    ):
        """
        img: (1, T, H, W). Each output frame is the denoised centre frame of a temporal window.
        Returns (1, T, H, W) on CPU (float).
        """
        t_max = img.shape[1]
        out = torch.zeros(1, t_max, img.shape[2], img.shape[3], dtype=torch.float32)
        for idx in tqdm(range(t_max), desc=desc):
            idxlist = subset_sampler(idx, t_max, self.frame_num)
            frames = img[:, idxlist, :, :].clone()
            den = self._mrc_tile_denoise_frame(
                frames, inner_net, shuffle_int, img_size, slice_pad, gain_value
            )
            out[:, idx, :, :] = den.squeeze(1)
        return out

    def configure_optimizers(self):
        optimizer1 = torch.optim.Adam(
            self.parameters(),   
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay
        )
        return optimizer1
                
    def loss_function(self, *args):
        if self.loss_F == 'L2':
            return nn.MSELoss(*args)
        elif self.loss_F == 'L1':
            return nn.L1Loss(*args)
        elif self.loss_F == 'SL1':
            return nn.SmoothL1Loss(*args)
        elif self.loss_F == 'BCE':
            return nn.BCEWithLogitsLoss()
        elif self.loss_F == 'FFL':
            return FocalFrequencyLoss()
        elif self.loss_F == 'NLL':
            return nn.PoissonNLLLoss(log_input=True)
        elif self.loss_F == 'charbonnier':
            return L1_Charbonnier_loss()
        elif self.loss_F == 'Mix':
            return MixedLoss()     

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/metric_1": 0, "hp/metric_2": 0})

    def train_dataloader(self):
        train_loader = DataLoader(self.Trainset, batch_size=self.batch_size, shuffle=True, pin_memory=True, 
                                    num_workers=8, drop_last=True, persistent_workers=True)
        return train_loader

    def val_dataloader(self):
        validation_loader = DataLoader(self.Validationset, batch_size=self.batch_size, shuffle=False, pin_memory=True, 
                                    num_workers=8, drop_last=False, persistent_workers=True)
        return validation_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.Testset, batch_size=1, shuffle=False, pin_memory=False, num_workers=4)
        return test_loader

    def predict_dataloader(self):
        test_loader = DataLoader(self.Testset, batch_size=1, shuffle=False, pin_memory=False, num_workers=4)
        return test_loader
    
    def training_step(self, batch, batch_nb):
        self.hfm = hfm(self.additional_dilation*2 + 2)
        if len(batch)==2:
            frames = batch[0]
            mask = batch[1]
        else:
            frames = batch
            mask = None
        if len(frames.shape)==3:
            frames = frames.unsqueeze(1)
        b,c,h,w = frames.shape
        now = frames[:,self.frame_num//2,:,:].unsqueeze(1)
        if c!=self.frame_num:
            frames = frames[:,:self.frame_num,:,:].squeeze(1).squeeze(2)
        out = self.forward(frames, 0) 
        if len(out)==2:
            out = out[0]
            mask = out[1]
        out_blind1_grad = self.forward(frames,shuffle = self.additional_dilation + 1)
        out_blind1 = out_blind1_grad.detach()
        if mask is not None:
            mask = mask.requires_grad_(False)
            out = out*(1-mask)
            now = now*(1-mask)
        if self.loss_F=='BCE':
            out = torch.sigmoid(out)
        else:
            loss = self.loss_function_forward(out,now)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        num_list = [0,1,2,3,4,5]
        if self.current_epoch%5==0 or self.current_epoch in num_list:
            if batch_nb == 0:
                sample_imgs = now[0,:,:,:]
                denoised_imgs = out[0,:,:,:]
                revisible_imgs = out_blind1[0,:,:,:]
                grid_sample = torchvision.utils.make_grid(sample_imgs, nrow=4, normalize=True, scale_each=True)
                grid_denoised = torchvision.utils.make_grid(denoised_imgs, nrow =4, normalize=True, scale_each=True)
                grid_revisible = torchvision.utils.make_grid(revisible_imgs, nrow=4, normalize=True, scale_each=True)
                self.logger.experiment.add_image('example_images', grid_sample, global_step = self.global_step)
                self.logger.experiment.add_image('denoised_image', grid_denoised, global_step = self.global_step)
                self.logger.experiment.add_image('revisible_image', grid_revisible, global_step = self.global_step)
        return loss

    def validation_step(self, batch, batch_nb):
        self.hfm = hfm(self.additional_dilation*2+2)
        if len(batch)==2:
            frames = batch[0]
            mask = batch[1]
        else:
            frames = batch
            mask= None
        if len(frames.shape)==3:
            frames = frames.unsqueeze(1)
        b,c,h,w = frames.shape
        now = frames[:,self.frame_num//2,:,:].unsqueeze(1)
        if c!=self.frame_num:
            frames = frames[:,:self.frame_num,:,:].squeeze(1).squeeze(2)
        out = self.forward(frames)
        if len(out)==2:
            out = out[0]
            mask = out[1]
        if mask is not None:
            out = out*(1-mask)
            now = now*(1-mask)
        if self.loss_F=='BCE':
            out = torch.sigmoid(out)
        else:
            with torch.no_grad():
                out_blind1 = self.forward(frames,shuffle=self.additional_dilation+1)
            val_loss = self.loss_function_forward(out,now)
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        num_list = [0,1,2,3,4,5]
        if self.current_epoch%5==0 or self.current_epoch in num_list:
            sample_imgs = now[0,:,:,:]
            denoised_imgs = out_blind1
            grid_sample = torchvision.utils.make_grid(sample_imgs, nrow=4, normalize=True, scale_each =True)
            grid_denoised = torchvision.utils.make_grid(denoised_imgs, nrow=4, normalize=True, scale_each=True)
            self.logger.experiment.add_image('val_example_images', grid_sample, global_step = self.global_step)
            self.logger.experiment.add_image('val_denoised_image', grid_denoised, global_step = self.global_step)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        img, idx, img_name, gain_value = batch
        img_size = 1024
        slice_pad = 64
        denoised_dir = self.save_folder+self.time_stamp+'/denoised'
        img_name = ''.join(img_name)
        if not os.path.exists(denoised_dir):
            os.makedirs(denoised_dir, exist_ok=True)
        mrc = mrcfile.new_mmap(os.path.join(denoised_dir, img_name), img.shape[-3:], mrc_mode=2, overwrite=True)
        if self.two_stage:
            conv_movie = self._denoise_full_mrc_movie(
                img,
                self.model,
                0,
                img_size,
                slice_pad,
                gain_value,
                'MRC stage1 (conventional)',
            )
            src2 = conv_movie.to(img.device)
            final_movie = self._denoise_full_mrc_movie(
                src2,
                self.second_model,
                0,
                img_size,
                slice_pad,
                gain_value,
                'MRC stage2 (annular)',
            )
            for t in range(final_movie.shape[1]):
                mrc.data[t, :, :] = np.array(final_movie[:, t, :, :].squeeze(), dtype=np.float32)
        else:
            final_movie = self._denoise_full_mrc_movie(
                img,
                self.model,
                0,
                img_size,
                slice_pad,
                gain_value,
                'Denoising Fraction',
            )
            for t in range(final_movie.shape[1]):
                mrc.data[t, :, :] = np.array(final_movie[:, t, :, :].squeeze(), dtype=np.float32)
        mrc.close()

    def _stitch_predict_from_normalized(
        self,
        frames_norm,
        inner_net,
        shuffle_int,
        img_size,
        slice_pad,
        ref_tensor,
    ):
        """frames_norm: (N, C, H, W) z-score normalized on CPU. Returns (N, 1, H, W) on CPU (model output)."""
        N, _, nH, nW = frames_norm.shape
        denoised = torch.zeros(N, 1, nH, nW, dtype=torch.float32)
        for i in range(0, nH, img_size):
            for j in range(0, nW, img_size):
                si = max(0, i - slice_pad)
                ei = min(nH, i + img_size + slice_pad)
                sj = max(0, j - slice_pad)
                ej = min(nW, j + img_size + slice_pad)
                xij = frames_norm[:, :, si:ei, sj:ej].to(ref_tensor.device).type_as(ref_tensor)
                yij = inner_net(xij, shuffle=shuffle_int).detach().cpu()
                top_pad = i - si
                left_pad = j - sj
                bottom_pad = ei - (i + img_size)
                right_pad = ej - (j + img_size)
                yij = yij[
                    :, :, top_pad : yij.shape[-2] - bottom_pad, left_pad : yij.shape[-1] - right_pad
                ]
                denoised[:, :, i : i + yij.shape[-2], j : j + yij.shape[-1]] = yij
        return denoised

    def predict_step(self, batch, batch_idx):
        frames, idx, img_name = batch
        img_size = 512
        slice_pad = 64
        frames_norm, nonsacle = numpy_zscore_normalize_test(frames)
        ref = frames
        shuffle1 = self.additional_dilation + 1
        shuffle2 = (
            self.second_additional_dilation + 1 if self.two_stage else shuffle1
        )

        if self.two_stage:
            den1 = self._stitch_predict_from_normalized(
                frames_norm, self.model, shuffle1, img_size, slice_pad, ref
            )
            # Single 5-frame crop: repeat stage-1 output across channels (no full movie).
            fr2 = den1.repeat(1, self.frame_num, 1, 1)
            fr2_norm, nonsacle2 = numpy_zscore_normalize_test(fr2)
            denoised = self._stitch_predict_from_normalized(
                fr2_norm, self.second_model, shuffle2, img_size, slice_pad, ref
            )
            denoised = numpy_zscore_recover(denoised, nonsacle2.cpu())
        else:
            now = frames[:, self.frame_num // 2, :, :].unsqueeze(1)
            denoised = torch.zeros_like(now).cpu()
            for i in range(0, frames_norm.size(2), img_size):
                for j in range(0, frames_norm.size(3), img_size):
                    si = max(0, i - slice_pad)
                    ei = min(frames_norm.size(2), i + img_size + slice_pad)
                    sj = max(0, j - slice_pad)
                    ej = min(frames_norm.size(3), j + img_size + slice_pad)
                    xij = frames_norm[:, :, si:ei, sj:ej]
                    yij = self.forward(xij.to(ref.device).type_as(ref), shuffle=shuffle1)
                    top_pad = i - si
                    left_pad = j - sj
                    bottom_pad = ei - (i + img_size)
                    right_pad = ej - (j + img_size)
                    yij = yij[
                        :,
                        :,
                        top_pad : yij.shape[-2] - bottom_pad,
                        left_pad : yij.shape[-1] - right_pad,
                    ]
                    denoised[:, :, i : i + yij.shape[-2], j : j + yij.shape[-1]] = yij.detach().cpu()
            denoised = numpy_zscore_recover(denoised, nonsacle.cpu())

        img_name = ''.join(img_name)
        img_name = os.path.splitext(img_name)[0] + '.tif'
        image = np.array(denoised.squeeze())
        denoised_dir = self.save_folder + self.time_stamp + '/denoised'
        if not os.path.exists(denoised_dir):
            os.makedirs(denoised_dir, exist_ok=True)
        cv.imwrite(os.path.join(denoised_dir, img_name), image.astype(np.float32))