import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_annular_spatial_mask(
    height: int,
    width: int,
    annulus_inner: int,
    annulus_outer: int,
    shuffle_true: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Returns a (1, 1, height, width) mask: 1 where convolution weights may be nonzero,
    0 at the kernel center (self-supervised blind pixel) and inside the annulus
    annulus_inner <= r <= annulus_outer (Euclidean distance on the integer grid).

    shuffle_true > 0 tightens the annulus toward the paper's "weaker blind spot"
    behaviour (smaller excluded ring), analogous to reducing dilation in the
    original ParamConv_reg_variable_dilation.
    """
    sh = int(max(0, shuffle_true))
    inner_r = annulus_inner + sh
    outer_r = max(annulus_inner, annulus_outer - sh)
    inner_sq = float(inner_r * inner_r)
    outer_sq = float(outer_r * outer_r)

    ci, cj = height // 2, width // 2
    ys = torch.arange(height, device=device, dtype=dtype).view(-1, 1) - ci
    xs = torch.arange(width, device=device, dtype=dtype).view(1, -1) - cj
    r_sq = ys * ys + xs * xs

    mask = torch.ones((1, 1, height, width), device=device, dtype=dtype)
    mask[:, :, ci, cj] = 0.0
    if inner_sq <= outer_sq:
        in_ring = (r_sq >= inner_sq) & (r_sq <= outer_sq)
        mask = mask * (~in_ring).to(dtype)
    return mask


class ParamConvAnnularMasked(nn.Module):
    """
    Depthwise-style masked conv: learnable weights only where the spatial mask is 1.
    Zeros the kernel center (standard blind-spot) and an annulus
    annulus_inner <= sqrt(dx^2+dy^2) <= annulus_outer in pixel offsets from the
    predicted pixel (kernel centre).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        padding,
        annulus_inner: int = 7,
        annulus_outer: int = 12,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.kernel_size_i = kernel_size[0]
        self.kernel_size_j = kernel_size[1]
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.annulus_inner = int(annulus_inner)
        self.annulus_outer = int(annulus_outer)
        if self.annulus_inner > self.annulus_outer:
            raise ValueError("annulus_inner must be <= annulus_outer")
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, self.kernel_size_i, self.kernel_size_j)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    @torch.autocast(device_type="cuda")
    def forward(self, x, shuffle_true=0):
        mask = _build_annular_spatial_mask(
            self.kernel_size_i,
            self.kernel_size_j,
            self.annulus_inner,
            self.annulus_outer,
            int(shuffle_true),
            x.device,
            x.dtype,
        )
        w = self.weight * mask
        pad_h, pad_w = self.padding[0], self.padding[1]
        x_pad = F.pad(x, [pad_w, pad_w, pad_h, pad_h])
        return F.conv2d(x_pad, w, self.bias, stride=self.stride, padding=0, dilation=1, groups=self.groups)


@torch.jit.script
def torch_zscore_normalize(image):
    new_image = torch.zeros_like(image).type_as(image)
    if len(image.shape)==2:
        v = image.flatten()
        new_image =(image-v.mean())/(v.std())
    if len(image.shape)==3:
        channels = image.shape[0]
        for c in range(channels):
            v = image[c,:,:].flatten()
            new_image[c,:,:] =(image[c,:,:]-v.mean())/(v.std())
    if len(image.shape)==4:
        batches = image.shape[0]
        channels = image.shape[1]
        for b in range(batches):
            for c in range(channels):
                v = image[b,c,:,:].flatten()
                new_image[b,c,:,:] =(image[b,c,:,:]-v.mean())/(v.std())
    return new_image  

class ParamConv_reg_variable_dilation(nn.Module):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False):
        super(ParamConv_reg_variable_dilation, self).__init__()

        self.kernel_size_i = kernel_size[0]
        self.kernel_size_j = kernel_size[1]
        # Create a static edge mask
        self.mask = torch.zeros(3, 3)
        self.mask[0,0]=1
        self.mask[1,0]=1
        self.mask[1,-1]=1
        self.mask[0,1]=1
        self.mask[-1,1]=1
        self.mask[0,-1]=1
        self.mask[-1,0]=1
        self.mask[-1,-1]=1

        # Identify non-zero positions in the mask
        self.indices = torch.nonzero(self.mask).t()
        # Number of parameters required
        num_params = 8  
        # Initialize the learnable parameters for the non-zero positions
        self.params = nn.Parameter(torch.Tensor(out_channels, in_channels, num_params))
        nn.init.kaiming_uniform_(self.params, a=math.sqrt(5))
        self.stride = stride
        self.padding = padding
        self.dilation = [self.kernel_size_i//2, self.kernel_size_j//2]
        self.groups = groups
        # Predefined zeroed kernel
        self.zero_kernel = torch.zeros(out_channels, in_channels, 3, 3).requires_grad_(False)

    @torch.autocast(device_type='cuda')
    def forward(self, x, shuffle_true=1):
        # Use the zeroed kernel and add learnable parameters
        kernel = self.zero_kernel.clone().to(x.device)
        dilation_real = [self.dilation[0] - shuffle_true, self.dilation[1] - shuffle_true]
        kernel[:,:,self.indices[0], self.indices[1]] = self.params
        padding = [self.padding[0] - shuffle_true, self.padding[1] - shuffle_true]
        if dilation_real[0] <= 0 or dilation_real[1] <= 0:
            dilation_real = [1,1]
            kernel = self.zero_kernel.clone().to(x.device)
            kernel[:,:,1,1] = torch.sum(self.params, dim=2)
            padding = [1,1]
        x_padding = F.pad(x, [padding[1], padding[1], padding[0], padding[0]])
        return F.conv2d(x_padding, kernel, bias=None, stride=self.stride, padding=0, dilation=dilation_real, groups=self.groups)


def _make_blind_param_conv(annulus, in_ch, out_ch, kernel_size, stride, padding, Bias):
    if annulus is None:
        return ParamConv_reg_variable_dilation(
            "donut", in_ch, out_ch, kernel_size, stride, padding, dilation=1, bias=Bias
        )
    return ParamConvAnnularMasked(
        in_ch,
        out_ch,
        kernel_size,
        stride,
        padding,
        annulus_inner=annulus[0],
        annulus_outer=annulus[1],
        bias=Bias,
    )


class SHINE(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        add_dilation=(0, 0),
        frame_num=5,
        filter=64,
        blocks=14,
        Bias=False,
        annulus=None,
    ):
        super(SHINE, self).__init__()
        self.n_frames = frame_num
        self.n_filters = filter
        self.n_block = blocks
        self.in_channels = in_channels
        self.add_dilation_i = add_dilation[0]
        self.add_dilation_j = add_dilation[1]
        self.annulus = annulus
        self.pad_before_downsample = nn.ModuleList()
        self.activation = nn.GELU()
        annulus_outer = annulus[1] if annulus is not None else 0

        def ensure_annular_kernel(ki, kj):
            if annulus_outer <= 0:
                return ki, kj
            m = 2 * annulus_outer + 1
            return max(ki, m), max(kj, m)

        ratio_1 = 3/4
        if frame_num == 1:
            ratio_1 = 1
        else:
            self.convolution_layer0_a = nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels * (self.n_frames - 1),
                            int(self.n_filters * (1 - ratio_1)),
                            3,
                            1,
                            1,
                            bias=Bias,
                        ),
                        self.activation,
                    )
                ]
                + [
                    nn.Sequential(
                        nn.Conv2d(
                            int(self.n_filters * (1 - ratio_1)),
                            int(self.n_filters * (1 - ratio_1)),
                            3,
                            1,
                            1,
                            bias=Bias,
                        ),
                        self.activation,
                    )
                    for _ in range(np.maximum(self.add_dilation_i, self.add_dilation_j))
                ]
            )
        self.convolution_layer0_b = nn.Conv2d(in_channels, int(self.n_filters * ratio_1), 1, 1, 0, bias=Bias)
        k0i = (self.add_dilation_i + 1) * 2 + 1
        k0j = (self.add_dilation_j + 1) * 2 + 1
        k0i, k0j = ensure_annular_kernel(k0i, k0j)
        p0i, p0j = k0i // 2, k0j // 2
        self.first_output = _make_blind_param_conv(
            annulus, self.n_filters, self.n_filters, (k0i, k0j), 1, (p0i, p0j), Bias
        )
        self.convolution_layer10 = torch.nn.ModuleList()
        self.convolution_layer2 = torch.nn.ModuleList()

        for i in range(4):
            if i > 0:
                scaler = 1
            elif i > -1:
                scaler = 1
            else:
                scaler = 1
            self.convolution_layer10.append(
                nn.Sequential(
                    nn.Conv2d(self.n_filters * scaler, self.n_filters * scaler, 3, 1, 1, bias=Bias),
                    self.activation,
                    nn.Conv2d(self.n_filters * scaler, self.n_filters * scaler, 3, 1, 1, bias=Bias),
                )
            )

            self.convolution_layer2.append(
                nn.Sequential(
                    nn.Conv2d(self.n_filters * scaler, self.n_filters * scaler, 3, 1, 1, bias=Bias),
                    self.activation,
                    nn.Conv2d(self.n_filters * scaler, self.n_filters * scaler, 3, 1, 1, bias=Bias),
                )
            )

        self.dilated_convs = nn.ModuleList()
        self.dilated_convs_res = nn.ModuleList()
        kernel_size_i = (self.add_dilation_i + 1 + 4) * 2 + 1
        kernel_size_j = (self.add_dilation_j + 1 + 4) * 2 + 1
        kernel_size_i, kernel_size_j = ensure_annular_kernel(kernel_size_i, kernel_size_j)
        padding_i = kernel_size_i // 2
        padding_j = kernel_size_j // 2
        for i in range(4):  # Create 5 dilated conv layers
            if i > 0:
                scaler = 1
            elif i > -1:
                scaler = 1
            else:
                scaler = 1
            layer = _make_blind_param_conv(
                annulus,
                self.n_filters * scaler,
                self.n_filters * scaler,
                (kernel_size_i, kernel_size_j),
                1,
                (padding_i, padding_j),
                Bias,
            )
            self.dilated_convs.append(layer)
            kri = max(1, kernel_size_i - 4)
            krj = max(1, kernel_size_j - 4)
            kri, krj = ensure_annular_kernel(kri, krj)
            pri, prj = kri // 2, krj // 2
            layer = _make_blind_param_conv(
                annulus,
                self.n_filters * scaler,
                self.n_filters * scaler,
                (kri, krj),
                1,
                (pri, prj),
                Bias,
            )
            self.dilated_convs_res.append(layer)
            # Update kernel_size and padding for the next iteration
            kernel_size_i = (kernel_size_i // 4 + 2 + 4) * 2 + 1
            kernel_size_j = (kernel_size_j // 4 + 2 + 4) * 2 + 1
            kernel_size_i, kernel_size_j = ensure_annular_kernel(kernel_size_i, kernel_size_j)
            padding_i = kernel_size_i // 2
            padding_j = kernel_size_j // 2

        feature_size = int(self.n_filters*(576/64))

        self.outconvs = nn.Sequential(
                                    nn.Conv2d(feature_size,feature_size//2,1,1,0,bias=Bias),
                                    self.activation,
                                    nn.Conv2d(feature_size//2,self.n_filters,1,1,0,bias=Bias),
                                    self.activation)                                   
        self.last_out = nn.Conv2d(self.n_filters,out_channels,1,1,0,bias=Bias)

        self.pool2d = nn.AvgPool2d(2)
        self.upsample_list = nn.ModuleList()
        for i in range(4):
            self.upsample_list.append(nn.Sequential(
                nn.Upsample(scale_factor=2**(i), mode='bilinear'),
                nn.Conv2d(self.n_filters, self.n_filters,1, stride=1, padding=0, bias=Bias),
                self.activation))

    def forward(self, input_image, shuffle=0):
        N, C, nH, nW = input_image.shape
        if nH % 64 != 0 or nW % 64 != 0:
            input_image = F.pad(input_image, [0, 64 - nW % 64, 0, 64 - nH % 64], mode = 'constant')
        if shuffle !=0:
            target = input_image[:,self.n_frames//2,:,:].unsqueeze(1).clone()
        else:
            target = input_image[:,self.n_frames//2,:,:].unsqueeze(1)
        if self.n_frames > 1:
            features = torch.cat((input_image[:,:self.n_frames//2,:,:], input_image[:,self.n_frames//2+1:,:,:]), dim=1)
            target = self.activation(self.convolution_layer0_b(target))
            features = self.convolution_layer0_a(features)
            base = torch.cat((target, features), dim=1)
        else:
            base = self.activation(self.convolution_layer0_b(input_image))

        initial_base = base
        output = self.activation(self.first_output(base, shuffle))
        base = self.activation(self.convolution_layer10[0](base) + base)
        merged = self.activation(self.dilated_convs_res[0](base, shuffle))
        output = torch.cat((output,merged),dim=1)
        base = self.activation(self.convolution_layer2[0](base) + base)
        for i in range(4):
            merged = self.activation(self.dilated_convs[i](base, shuffle))
            merged = self.upsample_list[i](merged)
            output = torch.cat((output,merged), dim=1)

            if i < 3:
                if i == -1:
                    base = self.pool2d(torch.cat((base,initial_base),dim=1))
                else:
                    base = self.pool2d(base)
                initial_base = base
                base = self.activation(self.convolution_layer10[i+1](base) + base)
                merged = self.activation(self.dilated_convs_res[i+1](base, shuffle))
                merged = self.upsample_list[i+1](merged)
                output = torch.cat((output, merged), dim=1)
                base = self.activation(self.convolution_layer2[i+1](base) + base)
        output = self.outconvs(output)
        output = self.last_out(output)
      
        if nH % 64 != 0 or nW % 64 != 0:
            output = output[:, :, 0:nH, 0:nW]

        return output