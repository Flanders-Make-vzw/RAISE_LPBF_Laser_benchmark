import numpy as np
import h5py
from torch.utils.data import Dataset
import pytorchvideo.transforms as vid_tvt
import torchvision.transforms as tvt
import torch
import math


def norm_uint8(x):
    return x / 255.0


class VideoCenterCrop(torch.nn.Module):
    """
    Transform for cropping video frames around the center.
    """

    def __init__(self, crop_size):
        super().__init__()

        self.crop_size = crop_size

    def forward(self, frames: torch.Tensor):
        height = frames.shape[2]
        width = frames.shape[3]
        y_offset = int(math.ceil((height - self.crop_size) / 2))
        x_offset = int(math.ceil((width - self.crop_size) / 2))

        return frames[:, :, y_offset: y_offset + self.crop_size, x_offset: x_offset + self.crop_size]


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self, slowfast_alpha):
        super().__init__()

        self.slowfast_alpha = slowfast_alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames, 1,
            torch.linspace(0, frames.shape[1] - 1,
                           frames.shape[1] // self.slowfast_alpha).long()
        )

        return [slow_pathway, fast_pathway]


class FramesSP(Dataset):
    def __init__(self, data_fp, test=False, repeat_frames=True, xoffset=20, yoffset=20, power_mean=215, speed_mean=900, **_):
        self.repeat_frames = repeat_frames
        self.xoffset, self.yoffset = xoffset, yoffset
        self.speed_mean, self.power_mean = speed_mean, power_mean
        self.data_fp = data_fp
        self.test = test

        # retrieving length
        with h5py.File(self.data_fp, 'r') as h5f:
            self.layer_cumsums = [np.cumsum([len(np.unique(h5f[o][l]['scan_line_index'][:]))
                                            for l in h5f[o].keys()]) for o in h5f.keys()]
            self.obj_cumsum = np.cumsum([o[-1] for o in self.layer_cumsums])
            self._len = self.obj_cumsum[-1]

    def __getitem__(self, index):
        # retrieve object, layer, scanline
        object_i = np.argwhere(self.obj_cumsum - index - 1 >= 0)[0][0]
        rindex = index if object_i == 0 else index - self.obj_cumsum[object_i - 1]
        layer_i = np.argwhere(self.layer_cumsums[object_i] - 1 - rindex >= 0)[0][0]
        rindex = rindex if layer_i == 0 else rindex - self.layer_cumsums[object_i][layer_i - 1]
        scan_line_i = rindex
        with h5py.File(self.data_fp, 'r') as h5f:  # open once ?
            object = list(h5f.keys())[object_i]
            layer = list(h5f[object].keys())[layer_i]
            indices = np.where(h5f[object][layer]['scan_line_index'][:] == scan_line_i)
            frames = h5f[object][layer]['frame'][indices]
            if not self.test:
                speed, power = h5f[object][layer]['laser_params'][scan_line_i]

        # crop frames around max intensity of mean frame
        i, j = np.unravel_index(frames.mean(0).argmax(), frames[0].shape)
        x = tvt.functional.crop(torch.tensor(np.array([frames])), i - self.xoffset, j - self.yoffset, 2 * self.xoffset + 1, 2 *
                                self.yoffset + 1)
        if self.repeat_frames:
            x = x.repeat_interleave(3, dim=0)

        if self.test:
            y = torch.tensor([None, None, object_i, layer_i, scan_line_i])
        else:
            y = torch.tensor([speed / self.speed_mean, power / self.power_mean, object_i, layer_i, scan_line_i])

        return x, y

    def __len__(self):
        return self._len


class OneWaySP(FramesSP):
    """
    Pytorch Dataset to interface the RAISE-LPBF-Laser benchmark data for one-way models like 3DResnet, X3D, MViT, etc.
    """

    def __init__(self, data_fp,
                 mean=0.45, std=0.225,
                 num_frames=8, crop_size=256, side_size=256, repeat_frames=True,
                 **kwargs):
        super().__init__(data_fp, repeat_frames=repeat_frames, **kwargs)

        _mean = [mean for _ in range(3)] if repeat_frames else [mean]
        _std = [std for _ in range(3)] if repeat_frames else [std]

        self.preprocess = tvt.Compose([
            vid_tvt.UniformTemporalSubsample(num_frames),
            tvt.Lambda(norm_uint8),
            vid_tvt.Normalize(_mean, _std),
            vid_tvt.ShortSideScale(size=side_size),
            VideoCenterCrop(crop_size)])

    def __getitem__(self, index):
        frames, y = super().__getitem__(index)

        x = self.preprocess(frames)

        return x, y


class TwoWaysSP(FramesSP):
    """
    Pytorch Dataset to interface the RAISE-LPBF-Laser benchmark data for two-way models like SlowFast.
    """

    def __init__(self, data_fp,
                 mean=0.45, std=0.225,
                 num_frames=32, crop_size=256, side_size=256, slowfast_alpha=4, repeat_frames=True,
                 **kwargs):
        super().__init__(data_fp, repeat_frames=repeat_frames, **kwargs)

        _mean = [mean for _ in range(3)] if repeat_frames else [mean]
        _std = [std for _ in range(3)] if repeat_frames else [std]

        self.preprocess = tvt.Compose([
            vid_tvt.UniformTemporalSubsample(num_frames),
            tvt.Lambda(norm_uint8),
            vid_tvt.Normalize(_mean, _std),
            vid_tvt.ShortSideScale(size=side_size),
            VideoCenterCrop(crop_size),
            PackPathway(slowfast_alpha)])

    def __getitem__(self, index):
        frames, y = super().__getitem__(index)

        x = self.preprocess(frames)

        return x, y
