import torch
import torch.nn as nn
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
    HIST_NUM_BINS,PARAM_HIDDEN_DIM
)

from model.filters import ContrastStretch, LocalKernel, DarkChannelDefog, Homomorphic, HistEq

IDENTITY_NOISE = 0.1 #std of the noise added to heads whose init is exactly identity (tone, kernel), so the filter output differs from x and its gate gets a real gradient


class ParamMaker(nn.Module): #turns the image embedding into the raw values each filter needs, one head per filter

    def __init__(self, emb, filters, hidden=PARAM_HIDDEN_DIM):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(emb, hidden), nn.ReLU()) #shared layer, shape: (B, emb) -> (B, hidden)
        self.heads = nn.ModuleDict() #one linear head per filter, keys are the filter index as a string

        for i, f in enumerate(filters):
            if f.n_params > 0: #HistEq has no parameters, so it gets no head
                head = nn.Linear(hidden, f.n_params) #shape: (B, hidden) -> (B, n_params)
                nn.init.zeros_(head.weight) #zero weights, so every image starts with the same raw values
                with torch.no_grad():
                    bias = f.init_bias().clone() #raw values that make the filter do nothing
                    if bool((bias == 0).all()): #tone and kernel start at exactly identity
                        bias += IDENTITY_NOISE * torch.randn_like(bias) #small noise so they start slightly off identity
                    head.bias.copy_(bias)

                self.heads[str(i)] = head 

    def forward(self, e):
        #e shape: (B, emb)
        z = self.trunk(e) #shape: (B, hidden)

        raw_values = {} #filter index -> raw values, shape: (B, n_params)
        for key, head in self.heads.items():
            i = int(key) #back to the filter index
            raw_values[i] = head(z)

        return raw_values