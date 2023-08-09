import torch
from torch.nn import Sequential, ReLU, Linear, \
    Module, Identity


class CNN3DResnet(Module):
    def __init__(self, fc_type='original', retrain_depth=0, fc_depth=5, weights_fp=None, strict_weights=True, **kwargs):
        super(CNN3DResnet, self).__init__()
        self.fc_depth = fc_depth

        self.cnn = torch.hub.load('facebookresearch/pytorchvideo:main', 'slow_r50', pretrained=True)
        for param in self.cnn.parameters():
            param.requires_grad = False
        if retrain_depth > 0:
            for param in self.cnn.blocks[-retrain_depth:].parameters():
                param.requires_grad = True
        elif retrain_depth == -1:
            for param in self.cnn.blocks.parameters():
                param.requires_grad = True

        if fc_type == 'single':
            self.cnn.blocks[-1].proj = Linear(2048, 1)
            self.linear_layers = Identity()
        elif fc_type == 'original':
            self.linear_layers = Sequential(
                Linear(400, 200),
                ReLU(inplace=True),
                *(l for _ in range(self.fc_depth) for l in (Linear(200, 200), ReLU(inplace=True))),
                Linear(200, 2),
            )

        if weights_fp is not None:
            self.load_weights(weights_fp, strict=strict_weights)

    def forward(self, x):
        x = self.cnn(x)
        x = self.linear_layers(x)
        return x

    def load_weights(self, fp, strict=True):
        self.load_state_dict(torch.load(fp), strict=strict)
