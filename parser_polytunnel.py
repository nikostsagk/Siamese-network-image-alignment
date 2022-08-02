import itertools
import os
import random
from glob import glob

import kornia.geometry.transform.flips as Kflips
import numpy as np
import torch as t
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image

from utils import plot_displacement2, plot_samples


class ImgPairDataset(Dataset):
    """
        Dataset structure:
        - collection_name
            - <session1 (hash key)>
                - <Images.png>
                - <Images.png>
            - <session2 (hash key)>
                - <Images.png>
                - <Images.png>
    """

    def __init__(self, path="/home/nikos/shared/data_collection/teach_and_repeat_2022_70cm"):
        super(ImgPairDataset, self).__init__()
        self.width = 512
        self.height = 384

        lvl1_subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
        all_session_images_pths = {}
        for subfolder in lvl1_subfolders:
            files = glob(subfolder + '/*.png', recursive=True)
            all_session_images_pths[subfolder] = files
            print(f"{subfolder} : {len(files)} images found")

        valid_columns = np.arange(0, 34) # 0 - 33 | No.34
        column_pairs = []
        for i in valid_columns:
            column_pairs.append(((i, len(valid_columns) - 1 - i)))
        # Remove bad examples. after c30_S, c3_N images become weird
        # Exiting the tunnel is not very robust
        column_pairs = column_pairs[:31]

        valid_tunnels = ["t0", "t1"]
        valid_rows = ["r0", "r1", "r2", "r3", "r4"]
        combinations = []
        for c in column_pairs:
            per_column_combinations = []
            for session in all_session_images_pths.keys():
                for r in valid_rows:
                    for t in valid_tunnels:
                        per_column_combinations.append(f"{session}/{t}-{r}-c{c[0]}_S.png")
                        per_column_combinations.append(f"{session}/{t}-{r}-c{c[1]}_N.png")
            combinations.append(per_column_combinations)

        self.data = []
        for c in combinations:
            per_column_pairs =[i for i in itertools.combinations(c, 2)]
            for pcp in per_column_pairs:
                self.data.append(pcp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if random.random() > 0.5:
            a, b = 0, 1
        else:
            b, a = 0, 1


        source_img = read_image(self.data[idx][a])/255.0
        target_img = read_image(self.data[idx][b])/255.0
        return source_img, target_img

class RiseholmePolytunnel(ImgPairDataset):

    def __init__(self, crop_width, fraction, smoothness, path="/home/nikos/shared/data_collection/teach_and_repeat_2022_70cm", eval=False):
        super(RiseholmePolytunnel, self).__init__(path=path)
        self.crop_width = crop_width
        self.fraction = fraction
        self.smoothness = smoothness
        self.eval = eval
        self.flip = Kflips.Hflip()

    def __getitem__(self, idx):
        source, target = super(RiseholmePolytunnel, self).__getitem__(idx)
        if self.eval:
            # Generate fake displacement of source image
            target, displacement = self.displace_img(target)
            return source, target, displacement

        source, target = self.augment(source, target)

        cropped_target, crop_start = self.crop_img(target)
        if self.smoothness == -1:
            heatmap = self.get_gaussian_heatmap(crop_start)
        elif self.smoothness == 0:
            heatmap = self.get_heatmap(crop_start)
        else:
            heatmap = self.get_smooth_heatmap(crop_start)
        return source, cropped_target, heatmap

    def displace_img(self, img):
        perc = 0.3
        displ = random.randint(int(-perc/2*self.width), int(perc/2*self.width))
        return t.roll(img, displ, dims=2), displ

    def crop_img(self, img):
        crop_start = random.randint(0, self.width - self.crop_width)
        return img[:, :, crop_start:crop_start + self.crop_width], crop_start

    def get_heatmap(self, crop_start):
        frac = self.width // self.fraction
        heatmap = t.zeros(frac)
        idx = int((crop_start + self.crop_width//2) / self.fraction)
        heatmap[idx] = 1
        return heatmap

    def get_gaussian_heatmap(self, crop_start):
        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

        frac = self.width // self.fraction

        heatmap = t.zeros(frac)
        mean_crop_idx = (crop_start + self.crop_width/2)
        mean_crop_frac = mean_crop_idx / self.fraction
        for i in range(frac):
            x = self.fraction * (i + 0.5)
            heatmap[i] = gaussian(x, mean_crop_idx, self.crop_width / 8)
        return heatmap

    def get_smooth_heatmap(self, crop_start):
        surround = self.smoothness * 2
        frac = self.width // self.fraction

        heatmap = t.zeros(frac + surround)
        idx = int((crop_start + self.crop_width//2) / self.fraction) + self.smoothness
        heatmap[idx] = 1
        idxs = np.array([-1, +1])
        for i in range(1, self.smoothness + 1):
            indexes = list(idx + i * idxs)
            for j in indexes:
                if 0 <= j < heatmap.size(0):
                    heatmap[j] = 1 - i * (1/(self.smoothness + 1))
        return heatmap[surround//2:-surround//2]

    def augment(self, source, target):
        source = transforms.Resize(self.height)(source)
        target = transforms.Resize(self.height)(target)
        if random.random() > 0.8:
            target = source.clone()
        if random.random() > 0.5:
            source = self.flip(source)
            target = self.flip(target)
        return source.squeeze(0), target



if __name__ == '__main__':
    eval = True
    #data = ImgPairDataset()
    data = RiseholmePolytunnel(248, 8, 3, eval=eval)

    print(f"Total image pairs: {len(data)}")

    idx = np.random.randint(0, len(data)+1)
    a, b, heatmap = data[idx]
    if not eval:
        plot_samples(a, b, heatmap, name="example")
    else:
        plot_displacement2(a, b, heatmap)

    # for idx in range(len(data)):
    #     a, b, heatmap = data[idx]
    #     name_a = data.data[idx][0].split("/")[-1].split(".png")[0]
    #     name_b = data.data[idx][1].split("/")[-1].split(".png")[0]
    #     plot_samples(a, b, heatmap, name=f"{name_a}-{name_b}", dir="/home/nikos/shared/data_collection/plot_samples/")
    # print(data.data[idx][0])
    # print(data.data[idx][1])

    #plot_img_pair(*data[idx])
