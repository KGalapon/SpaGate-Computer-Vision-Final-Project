import torch
import torch.nn as nn

from model.filters import ContrastStretch, LocalKernel, DarkChannelDefog, Homomorphic, HistEq, ResidualNet, AODFilter

from model.encoder import Encoder
from model.param_maker import ParamMaker
from model.gate import Gate

from config import EMB, GATE_SCALE, GATE_DETACH_FILTER, STAGE_FILTERS, DEFOG_MAX_OMEGA


FILTER_CLASSES = {
    "tone": ContrastStretch,
    "kernel": LocalKernel,
    "defog": DarkChannelDefog,
    "homomorphic": Homomorphic,
    "histeq": HistEq,
    "residual": ResidualNet,
    "aod": AODFilter,
}


class Stage(nn.Module): #one block: its own encoder, runs its filters in parallel on its input, gates each one per pixel, and blends them

    def __init__(self, names, emb, gate_scale, detach_filter):
        super().__init__()
        for name in names:
            assert name in FILTER_CLASSES, f"unknown filter {name}, choose from {list(FILTER_CLASSES)}"
        self.names = list(names) #filter names of this stage
        self.filters = nn.ModuleList([FILTER_CLASSES[name]() for name in names])
        self.enc = Encoder(emb) #looks at this stage's input image, so stage 2 sees what stage 1 produced
        self.pm = ParamMaker(emb, self.filters) #embedding -> raw values for each filter of this stage

        self.gates = nn.ModuleList() #one gate per filter
        for _ in names:
            self.gates.append(Gate(emb, scale=gate_scale, detach_filter=detach_filter))

    def forward(self, x):
        #x shape: (B, 3, H, W)
        e, feat = self.enc(x, return_features=True) #shapes: (B, emb) and (B, C, 8, 8)
        raw_values = self.pm(e) #dict, filter index -> raw values, HistEq has no entry

        outs = [] #output of every filter, each shape: (B, 3, H, W)
        for i, f in enumerate(self.filters):
            outs.append(f(x, raw_values.get(i))) #None for HistEq

        gate_list = [] #one gate per filter, each shape: (B, 1, H, W)
        for gate, t in zip(self.gates, outs):
            gate_list.append(gate(x, t, e, feat))
        G = torch.cat(gate_list, dim=1) #shape: (B, n_filters, H, W)

        G_sum = G.sum(dim=1, keepdim=True) #total gate at each pixel
        G_sum = torch.clamp(G_sum, min=1.0) #only shrink the gates where they add up to more than 1
        Gn = G / G_sum #normalised gates, so the update stays convex

        out = x #start from this stage's input
        for i in range(len(outs)):
            change = outs[i] - x #how far filter i moves each pixel
            out = out + Gn[:, i:i + 1] * change #add that change, weighted by its gate

        return out.clamp(0, 1), G #G is returned before normalising


class SpaGate(nn.Module): #stages run one after another, inside a stage the filters run in parallel and are gated per pixel

    def __init__(self, emb=EMB, gate_scale=GATE_SCALE, detach_filter=GATE_DETACH_FILTER, stage_filters=STAGE_FILTERS):
        super().__init__()
        self.stages = nn.ModuleList([Stage(names, emb, gate_scale, detach_filter) for names in stage_filters])

        self.gate_names = [] #one label per gate, in the same order as the channels of G, for example s1:tone
        for s, names in enumerate(stage_filters):
            for name in names:
                self.gate_names.append(f"s{s + 1}:{name}")

    def forward(self, x):
        #x shape: (B, 3, H, W)
        current = x
        gate_maps = [] #G of every stage
        for stage in self.stages:
            current, G = stage(current) #the next stage sees the enhanced image
            gate_maps.append(G)
        return current, torch.cat(gate_maps, dim=1) #shapes: (B, 3, H, W) and (B, total number of gates, H, W)

    def gate_modules(self):
        return [gate for stage in self.stages for gate in stage.gates] #flat list, same order as gate_names

    def gate_parameters(self):
        return [p for gate in self.gate_modules() for p in gate.parameters()]

    @torch.no_grad()
    def defog_omega(self, x):
        #mean omega the first stage's defog filter predicts for these images, nan if there is no defog filter
        stage = self.stages[0]
        for i, name in enumerate(stage.names):
            if name == "defog":
                e = stage.enc(x)
                raw = stage.pm(e)[i]
                return (DEFOG_MAX_OMEGA * torch.sigmoid(raw)).mean().item()
        return float("nan")