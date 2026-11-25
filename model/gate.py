import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EMB, GATE_CHANNELS, GATE_SCALE, GATE_DETACH_FILTER, GATE_USE_FEATURES, ENCODER_CHANNELS

GATE_LOGIT_LIMIT = 4.0 #the gate logit is squashed to [-4, 4], so the gate stays in about [0.018, 0.982] and never saturates or dies
GATE_INIT_PROB = 0.2 #every gate starts at this value, so the normalisation does nothing at the start


class Gate(nn.Module): #decides per pixel how much of a filter's output to use, from [x, t, t - x], the encoder feature map and the embedding

    def __init__(self, emb=EMB, ch=GATE_CHANNELS, scale=GATE_SCALE, detach_filter=GATE_DETACH_FILTER, use_features=GATE_USE_FEATURES):
        super().__init__()
        self.ch = ch #number of feature maps in the hidden convs
        self.scale = scale #the gate is computed at this fraction of the image size
        self.detach_filter = detach_filter #if true, no gradient flows from the gate back into the filter

        self.use_features = use_features #if true, the encoder's spatial feature map is an extra input
        first_in = 9 + (ENCODER_CHANNELS[-1] if use_features else 0) #9 = x, t, t - x, plus the encoder feature channels
        conv_in_channels = [first_in, ch, ch] #input channels of the three convs
        self.convs = nn.ModuleList() #three 3x3 convs
        self.films = nn.ModuleList() #one film layer per conv, turns the embedding into film_g and film_b

        for in_channels in conv_in_channels:
            self.convs.append(nn.Conv2d(in_channels, ch, kernel_size=3, padding=1))

            film = nn.Linear(emb, 2 * ch) #shape: (B, emb) -> (B, 2 * ch), first half is film_g, second half film_b
            nn.init.zeros_(film.weight) #zero weights and bias, so film_g = 0 and film_b = 0
            nn.init.zeros_(film.bias) #and the film layer changes nothing at the start
            self.films.append(film)

        self.out = nn.Conv2d(ch, 1, kernel_size=3, padding=1) #one gate logit per pixel
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, math.log(GATE_INIT_PROB / (1 - GATE_INIT_PROB))) #logit of GATE_INIT_PROB, sigmoid has no dead zone so the gate can always recover

    def forward(self, x, t, e, feat=None):
        #x: image, t: the filter's output, both shape: (B, 3, H, W), e: embedding, shape: (B, emb), feat: encoder feature map, shape: (B, C, fh, fw)
        batch_size = x.shape[0] #number of images
        height, width = x.shape[-2:] #full size, needed to upsample the gate again

        if self.detach_filter:
            t = t.detach() #the gate does not send gradient into the filter

        h = torch.cat([x, t, t - x], dim=1) #shape: (B, 9, H, W)

        if self.scale != 1:
            h = F.interpolate(h, scale_factor=self.scale, mode="bilinear", align_corners=False) #downsample

        if self.use_features:
            feat_up = F.interpolate(feat, size=h.shape[-2:], mode="bilinear", align_corners=False) #stretch the coarse feature map to the gate's resolution
            h = torch.cat([h, feat_up], dim=1) #shape: (B, 9 + C, h, w), every pixel now also knows what its region looks like

        for conv, film in zip(self.convs, self.films):
            film_values = film(e) #shape: (B, 2 * ch)
            film_g, film_b = film_values.chunk(2, dim=1) #each shape: (B, ch)
            film_g = film_g.view(batch_size, self.ch, 1, 1) #so they broadcast over the image
            film_b = film_b.view(batch_size, self.ch, 1, 1)

            h = conv(h) #shape: (B, ch, h, w)
            h = h * (1 + film_g) + film_b #the embedding scales and shifts every feature map
            h = F.relu(h)

        logit = GATE_LOGIT_LIMIT * torch.tanh(self.out(h) / GATE_LOGIT_LIMIT) #soft clamp of the logit, close to the identity near 0
        g = torch.sigmoid(logit) #shape: (B, 1, h, w), gradient stays away from 0

        if self.scale != 1:
            g = F.interpolate(g, size=(height, width), mode="bilinear", align_corners=False) #back to full size

        return g #the gate, shape: (B, 1, H, W)