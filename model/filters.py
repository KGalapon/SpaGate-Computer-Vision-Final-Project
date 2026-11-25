import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CONTRAST_NUM_SEGMENTS,
    KERNEL_SIZE, KERNEL_SCALE,
    DEFOG_PATCH_SIZE, DEFOG_MIN_TRANSMISSION, DEFOG_MIN_AIRLIGHT, DEFOG_TOP_FRACTION,
    DEFOG_MAX_OMEGA, DEFOG_INIT_RAW,
    HOMO_PAD, HOMO_CUTOFF_RANGE, HOMO_GAIN_LOW_RANGE, HOMO_GAIN_HIGH_RANGE,
    HOMO_INIT_CUTOFF, HOMO_INIT_GAIN_LOW, HOMO_INIT_GAIN_HIGH,
    HIST_NUM_BINS,
    RESIDUAL_CHANNELS, RESIDUAL_SCALE,
    AOD_CHANNELS, AOD_K_MAX, AOD_B_RANGE,
)


def luminance(image):
    red = image[:, 0:1, :, :] #red channel, shape: (B, 1, H, W)
    green = image[:, 1:2, :, :] #green channel
    blue = image[:, 2:3, :, :] #blue channel
    return 0.27 * red + 0.67 * green + 0.06 * blue #weighted sum, shape: (B, 1, H, W)


def squash_to_range(raw_value, low, high):
    return low + (high - low) * torch.sigmoid(raw_value) #any real number becomes a number between low and high


def raw_value_for_target(target, low, high):
    fraction = (target - low) / (high - low) #where target sits between low and high, 0 to 1
    return math.log(fraction / (1 - fraction)) #inverse of the sigmoid, gives the raw value that produces target


class ContrastStretch(nn.Module): #tone curve, one curve per colour channel
    n_params = 3 * CONTRAST_NUM_SEGMENTS #three channels, one logit per segment

    def __init__(self):
        super().__init__()
        input_knots = torch.linspace(0, 1, CONTRAST_NUM_SEGMENTS + 1) #divide [0,1] into equal parts
        self.register_buffer("r", input_knots) #r values, buffer so it moves to the gpu with the module

    def init_bias(self):
        return torch.zeros(self.n_params) #equal heights, so s equals r and the curve is a straight line

    def output_knots(self, raw):
        batch_size = raw.shape[0] #number of images
        logits = raw.view(batch_size, 3, CONTRAST_NUM_SEGMENTS) #shape: (B, 3, num_segments)

        heights = torch.softmax(logits, dim=-1) #positive and sum to 1
        s_after_start = torch.cumsum(heights, dim=-1) #running total, s_1 to the last s, last one is 1

        zero_column = torch.zeros(batch_size, 3, 1, device=raw.device, dtype=raw.dtype) #s_0 is always 0
        s = torch.cat([zero_column, s_after_start], dim=-1) #shape: (B, 3, num_segments + 1), starts at 0 and ends at 1
        return s

    def channel_contrast_stretch(self, x_channel, s_values):
        #x_channel shape: (B, H, W), s_values shape: (B, num_segments + 1) for this channel
        batch_size = x_channel.shape[0] #number of images
        result = torch.zeros_like(x_channel) #output, filled in one segment at a time

        for k in range(CONTRAST_NUM_SEGMENTS):
            r_left = self.r[k] #start of segment k on the input axis
            r_right = self.r[k + 1] #end of segment k
            s_left = s_values[:, k].view(batch_size, 1, 1) #output value at the start of the segment
            s_right = s_values[:, k + 1].view(batch_size, 1, 1) #output value at the end of the segment


            # formula for the segment
            slope = (s_right - s_left) / (r_right - r_left) #how far across the segment the pixel is, 0 to 1
            value = s_left + slope * (x_channel - r_left)  #straight line between the two knots

            if k < CONTRAST_NUM_SEGMENTS - 1:
                in_segment = (x_channel >= r_left) & (x_channel < r_right) #pixels that fall in this segment
            else:
                in_segment = x_channel >= r_left #last segment 

            result = torch.where(in_segment, value, result) #keep the new value only for pixels in this segment

        return result

    def forward(self, x, raw):
        s = self.output_knots(raw) #shape: (B, 3, num_segments + 1)

        red = self.channel_contrast_stretch(x[:, 0], s[:, 0]) #red channel with its own curve
        green = self.channel_contrast_stretch(x[:, 1], s[:, 1]) #green channel
        blue = self.channel_contrast_stretch(x[:, 2], s[:, 2]) #blue channel

        return torch.stack([red, green, blue], dim=1) #back to shape: (B, 3, H, W)




class LocalKernel(nn.Module): #local filter, output = x * conv(x, kernel_1) + conv(x, kernel_2) + x
    n_params = 2 * 3 * KERNEL_SIZE * KERNEL_SIZE #two kernels, three channels, kernel_size ** 2 values each

    def init_bias(self):
        return torch.zeros(self.n_params) #tanh(0) = 0, kernels are all zero, output equals x

    def make_kernels(self, raw):
        batch_size = raw.shape[0] #number of images

        kernels = torch.tanh(raw) #limit to [-1, 1]
        kernels = kernels * KERNEL_SCALE #then shrink so the filter cannot blow up
        kernels = kernels.view(batch_size, 2, 3, KERNEL_SIZE, KERNEL_SIZE) #shape: (B, which kernel, channel, k, k)
        return kernels

    def channel_local_filter(self, x_channel, kernel_1, kernel_2):
        #x_channel shape: (B, H, W), kernel_1 and kernel_2 shape: (B, k, k)
        batch_size = x_channel.shape[0] #number of images
        half = KERNEL_SIZE // 2 #how many pixels to pad on each side

        outputs = [] #one filtered image per entry
        for b in range(batch_size):
            image = x_channel[b].unsqueeze(0).unsqueeze(0) #shape: (1, 1, H, W)
            padded = F.pad(image, [half, half, half, half], mode="reflect") #pad so the output keeps the same size

            weights_1 = kernel_1[b].view(1, 1, KERNEL_SIZE, KERNEL_SIZE) #conv2d wants (out, in, h, w)
            weights_2 = kernel_2[b].view(1, 1, KERNEL_SIZE, KERNEL_SIZE)

            response_1 = F.conv2d(padded, weights_1) #shape: (1, 1, H, W)
            response_2 = F.conv2d(padded, weights_2) #shape: (1, 1, H, W)

            filtered = image * response_1 + response_2 + image #the filter formula
            outputs.append(filtered[0, 0]) #drop the extra dimensions, shape: (H, W)

        return torch.stack(outputs, dim=0) #shape: (B, H, W)

    def forward(self, x, raw):
        kernels = self.make_kernels(raw) #shape: (B, 2, 3, k, k)

        red = self.channel_local_filter(x[:, 0], kernels[:, 0, 0], kernels[:, 1, 0]) #both red kernels
        green = self.channel_local_filter(x[:, 1], kernels[:, 0, 1], kernels[:, 1, 1]) #both green kernels
        blue = self.channel_local_filter(x[:, 2], kernels[:, 0, 2], kernels[:, 1, 2]) #both blue kernels

        result = torch.stack([red, green, blue], dim=1) #back to shape: (B, 3, H, W)
        return torch.clamp(result, 0, 1) #keep pixel values valid


class DarkChannelDefog(nn.Module): #removes fog, I = J * T + A * (1 - T), omega is the one learned value
    n_params = 1 #just omega

    def init_bias(self):
        return torch.tensor([DEFOG_INIT_RAW]) #very negative raw value, omega is near 0, almost no fog removed

    def dark(self, x):
        #x shape: (B, 3, H, W)
        x_min = torch.min(x, dim=1, keepdim=True).values #x.min(1), high when there is a lot of fog, shape: (B, 1, H, W)
        padding = DEFOG_PATCH_SIZE // 2 #pad so the output keeps the same size

        negated = -x_min #a min-pool is a max-pool on the negated values
        pooled = F.max_pool2d(negated, kernel_size=DEFOG_PATCH_SIZE, stride=1, padding=padding)
        return -pooled #negate back, smallest value in each patch, this is the dark channel

    def estimate_A(self, I, fog_mask):
        batch_size, channels, height, width = I.shape #image sizes
        num_top_pixels = max(1, int(DEFOG_TOP_FRACTION * height * width)) #top 0.1% of the fog mask values

        A_list = [] #one rgb colour per image
        for b in range(batch_size):
            fog_mask_flat = fog_mask[b].flatten() #shape: (H*W,)
            top_indices = torch.topk(fog_mask_flat, num_top_pixels).indices #positions of the foggiest pixels

            I_flat = I[b].reshape(3, height * width) #shape: (3, H*W)
            top_pixels = I_flat[:, top_indices] #colours of those pixels, shape: (3, num_top_pixels)
            A_list.append(top_pixels.mean(dim=1)) #average colour, shape: (3,)

        A = torch.stack(A_list, dim=0) #shape: (B, 3)
        A = A.view(batch_size, 3, 1, 1) #so it broadcasts over the image
        A = torch.clamp(A, min=DEFOG_MIN_AIRLIGHT) #stops the division by A from blowing up
        return A.detach() #A is not learned through backprop

    def estimate_T(self, I, A, omega):
        I_over_A = I / A #I/A
        dark_I_over_A = self.dark(I_over_A) #dark(I/A), equals 1 - T for a clear scene
        T = 1 - omega * dark_I_over_A #T controls the mixing, bigger omega means more fog removed
        return torch.clamp(T, min=DEFOG_MIN_TRANSMISSION) #stops the division by T from blowing up

    def forward(self, I, raw):
        batch_size = I.shape[0] #number of images

        omega = DEFOG_MAX_OMEGA * torch.sigmoid(raw) #between 0 and the max omega
        omega = omega.view(batch_size, 1, 1, 1) #so it broadcasts over the image

        fog_mask = self.dark(I) #dark channel, shape: (B, 1, H, W)
        A = self.estimate_A(I, fog_mask) #shape: (B, 3, 1, 1)
        T = self.estimate_T(I, A, omega) #shape: (B, 1, H, W)

        J = (I - A) / T + A #J = (I - A) / T + A, the clear scene
        return torch.clamp(J, 0, 1) #keep pixel values valid


class Homomorphic(nn.Module): #filters the log of the luminance, learned values: cutoff, gain_low, gain_high
    n_params = 3 #cutoff, gain_low, gain_high
    EPS = 1e-3 #stops log(0) and division by zero

    def init_bias(self):
        cutoff_raw = raw_value_for_target(HOMO_INIT_CUTOFF, *HOMO_CUTOFF_RANGE) #raw value that gives the starting cutoff
        gain_low_raw = raw_value_for_target(HOMO_INIT_GAIN_LOW, *HOMO_GAIN_LOW_RANGE) #both gains start at 1.0
        gain_high_raw = raw_value_for_target(HOMO_INIT_GAIN_HIGH, *HOMO_GAIN_HIGH_RANGE) #so the filter is all ones and nothing changes
        return torch.tensor([cutoff_raw, gain_low_raw, gain_high_raw])

    def get_settings(self, raw):
        batch_size = raw.shape[0] #number of images

        cutoff = squash_to_range(raw[:, 0], *HOMO_CUTOFF_RANGE).view(batch_size, 1, 1, 1) #shape: (B, 1, 1, 1)
        gain_low = squash_to_range(raw[:, 1], *HOMO_GAIN_LOW_RANGE).view(batch_size, 1, 1, 1)
        gain_high = squash_to_range(raw[:, 2], *HOMO_GAIN_HIGH_RANGE).view(batch_size, 1, 1, 1)
        return cutoff, gain_low, gain_high

    def make_frequency_filter(self, padded_height, padded_width, cutoff, gain_low, gain_high, device):
        freq_y = torch.fft.fftfreq(padded_height, device=device).view(padded_height, 1) #vertical frequency of each fft row
        freq_x = torch.fft.fftfreq(padded_width, device=device).view(1, padded_width) #horizontal frequency of each fft column
        freq_squared = freq_x ** 2 + freq_y ** 2 #squared distance from frequency 0, shape: (Hp, Wp)

        high_pass = 1 - torch.exp(-freq_squared / (2 * cutoff ** 2)) #0 at frequency 0, rises to 1, shape: (B, 1, Hp, Wp)

        return gain_low + (gain_high - gain_low) * high_pass #gain_low for low frequencies, gain_high for high

    def filter_log_luminance(self, log_lum, cutoff, gain_low, gain_high):
        height, width = log_lum.shape[2], log_lum.shape[3] #original size

        padded = F.pad(log_lum, [HOMO_PAD, HOMO_PAD, HOMO_PAD, HOMO_PAD], mode="reflect") #so the fft does not wrap the edges around
        padded_height, padded_width = padded.shape[2], padded.shape[3] #size after padding

        freq_filter = self.make_frequency_filter(
            padded_height, padded_width, cutoff, gain_low, gain_high, log_lum.device
        ) #shape: (B, 1, Hp, Wp)

        spectrum = torch.fft.fft2(padded) #image to frequencies
        filtered_spectrum = spectrum * freq_filter #scale each frequency
        filtered = torch.fft.ifft2(filtered_spectrum).real #back to an image

        return filtered[:, :, HOMO_PAD:HOMO_PAD + height, HOMO_PAD:HOMO_PAD + width] #remove the padding again

    def forward(self, x, raw):
        cutoff, gain_low, gain_high = self.get_settings(raw) #the three learned values

        lum = luminance(x) #shape: (B, 1, H, W)
        log_lum = torch.log(lum + self.EPS) #work in log space

        filtered_log_lum = self.filter_log_luminance(log_lum, cutoff, gain_low, gain_high)

        new_lum = torch.exp(filtered_log_lum) #back from log space
        ratio = new_lum / (lum + self.EPS) #how much each pixel's luminance changed

        return torch.clamp(x * ratio, 0, 1) #apply the same ratio to r, g, b


class HistEq(nn.Module): #histogram equalisation on luminance, no learned parameters
    n_params = 0 #nothing to learn
    EPS = 1e-3 #stops division by zero

    def init_bias(self):
        return torch.zeros(0) #empty, no parameters

    def equalise_one_image(self, image):
        #image shape: (3, H, W)
        lum = luminance(image.unsqueeze(0))[0] #shape: (1, H, W)
        lum_flat = lum.flatten() #shape: (H*W,)
        num_pixels = lum_flat.numel() #total number of pixels

        histogram = torch.histc(lum_flat, bins=HIST_NUM_BINS, min=0, max=1) #pixels per luminance bin
        cdf = torch.cumsum(histogram, dim=0) / num_pixels #fraction of pixels at or below each bin, shape: (num_bins,)

        bin_index = (lum_flat * (HIST_NUM_BINS - 1)).long() #which bin each pixel falls in
        bin_index = torch.clamp(bin_index, 0, HIST_NUM_BINS - 1) #keep inside the valid bins
        new_lum = cdf[bin_index].view_as(lum) #new luminance is the cdf value, shape: (1, H, W)

        ratio = new_lum / (lum + self.EPS) #how much each pixel's luminance changed
        return image * ratio #apply the same ratio to r, g, b

    @torch.no_grad()
    def forward(self, x, raw=None):
        outputs = [] #one equalised image per entry
        for b in range(x.shape[0]):
            outputs.append(self.equalise_one_image(x[b]))

        result = torch.stack(outputs, dim=0) #shape: (B, 3, H, W)
        return torch.clamp(result, 0, 1) #keep pixel values valid


class ResidualNet(nn.Module): #small conv net that learns a per-pixel colour change, the other filters are global curves or one kernel, this one can do anything local
    n_params = RESIDUAL_CHANNELS #the parameter maker predicts one shift per hidden channel from the image embedding, that is the global context

    def __init__(self, ch=RESIDUAL_CHANNELS):
        super().__init__()
        self.ch = ch #width of the hidden layers
        self.conv1 = nn.Conv2d(3, ch, kernel_size=3, padding=1) #works at half resolution, so it is cheap
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=2, dilation=2) #dilation makes it see further
        self.conv3 = nn.Conv2d(ch, ch, kernel_size=3, padding=4, dilation=4)
        self.out = nn.Conv2d(ch, 3, kernel_size=3, padding=1) #the colour change
        nn.init.zeros_(self.out.weight) #zero weights and bias, so the change is 0 and the output equals x at the start
        nn.init.zeros_(self.out.bias)

    def init_bias(self):
        return torch.zeros(self.n_params) #no shift at the start

    def forward(self, x, raw):
        #x shape: (B, 3, H, W), raw shape: (B, ch)
        batch_size = x.shape[0] #number of images

        h = F.avg_pool2d(x, 2) #half resolution, shape: (B, 3, H/2, W/2)
        h = F.leaky_relu(self.conv1(h), 0.1) #shape: (B, ch, H/2, W/2)
        h = h + raw.view(batch_size, self.ch, 1, 1) #image-wide shift predicted from the embedding
        h = F.leaky_relu(self.conv2(h), 0.1)
        h = F.leaky_relu(self.conv3(h), 0.1)

        change = self.out(h) #shape: (B, 3, H/2, W/2)
        change = F.interpolate(change, size=x.shape[-2:], mode="bilinear", align_corners=False) #back to full size
        return torch.clamp(x + RESIDUAL_SCALE * torch.tanh(change), 0, 1) #keep pixel values valid


class AODFilter(nn.Module): #J = K(x) * x - K(x) + b, the fog model folded into one expression, a small multi-scale net predicts K for every pixel
    n_params = 3 + AOD_CHANNELS #3 values for b (one per colour) and one shift per hidden channel, both predicted from the image embedding

    def __init__(self, ch=AOD_CHANNELS):
        super().__init__()
        self.ch = ch #width of the hidden layers
        self.log_k_max = math.log(AOD_K_MAX)
        self.conv1 = nn.Conv2d(3, ch, kernel_size=1) #the four convs see the image at different scales
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(2 * ch, ch, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(2 * ch, ch, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(4 * ch, 3, kernel_size=3, padding=1) #sees all four, gives K for the three colours
        nn.init.zeros_(self.conv5.weight) #zero weights and bias, so K = 1 at the start
        nn.init.zeros_(self.conv5.bias)

    def init_bias(self):
        return torch.zeros(self.n_params) #b = 1 and no shift at the start, with K = 1 the output equals x

    def forward(self, x, raw):
        #x shape: (B, 3, H, W), raw shape: (B, 3 + ch)
        batch_size = x.shape[0] #number of images

        b = 1 + AOD_B_RANGE * torch.tanh(raw[:, :3]).view(batch_size, 3, 1, 1) #shape: (B, 3, 1, 1), 1 gives a white fog colour
        shift = raw[:, 3:].view(batch_size, self.ch, 1, 1) #image-wide context for the hidden layers

        h = F.avg_pool2d(x, 2) #half resolution, K is smooth like the fog, shape: (B, 3, H/2, W/2)
        x1 = F.relu(self.conv1(h))
        x2 = F.relu(self.conv2(x1) + shift)
        x3 = F.relu(self.conv3(torch.cat([x1, x2], dim=1)))
        x4 = F.relu(self.conv4(torch.cat([x2, x3], dim=1)))
        k_raw = self.conv5(torch.cat([x1, x2, x3, x4], dim=1)) #shape: (B, 3, H/2, W/2)

        k = torch.exp(self.log_k_max * torch.tanh(k_raw)) #positive, between 1 / AOD_K_MAX and AOD_K_MAX, equal to 1 when k_raw is 0
        k = F.interpolate(k, size=x.shape[-2:], mode="bilinear", align_corners=False) #back to full size
        return torch.clamp(k * x - k + b, 0, 1) #keep pixel values valid