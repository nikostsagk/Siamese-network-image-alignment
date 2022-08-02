import os

import numpy as np
import torch as t
from torchvision.io import read_image
from torchvision.transforms import Resize

from model import get_parametrized_model, load_model
from utils import interpolate_histogram, plot_displacement

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

MODEL_PATH = "./model_siam.pt"
#MODEL_PATH = "/home/nikos/shared/results_v1/models/siam_109.pt"
SOURCE_IMG_DIR = "/home/nikos/shared/data_collection/teach_and_repeat_2022_70cm/62da663663002c794c29b6cd/" # t*-r*-c15_S
TARGET_IMG = "/home/nikos/shared/data_collection/teach_and_repeat_2022_70cm/62da663663002c794c29b6cd/t1-r2-c0_S.png"
COLUMN_HEADING = TARGET_IMG.split("-")[-1].split(".png")[0]
# -------------------------------


WIDTH = 512
OUTPUT_SIZE = 64
FRACTION = int(WIDTH / OUTPUT_SIZE)
PAD = int((OUTPUT_SIZE - 2) / 2)

DISPLACEMENT = 0 # in pixels

def get_average_embedding():

    model = get_parametrized_model(False, 3, 256, 0, PAD, device)
    model = load_model(model, MODEL_PATH)

    model.eval()
    with t.no_grad():
        source_img_files = [os.path.join(SOURCE_IMG_DIR, f) for f in os.listdir(SOURCE_IMG_DIR) if COLUMN_HEADING in f]
        # Get img size
        img_shape = read_image(source_img_files[0]).numpy().shape
        transform = Resize(int(img_shape[1] * WIDTH / img_shape[2]))
        source_imgs = t.empty((len(source_img_files), 3, int(img_shape[1] * WIDTH / img_shape[2]), 512)).to(device)
        for n, img in enumerate(source_img_files):
            source_imgs[n] = transform(read_image(img) / 255.0).to(device)
        target = transform(read_image(TARGET_IMG) / 255.0).to(device)[..., FRACTION//2:-FRACTION//2]
        target_displaced = t.roll(target, -DISPLACEMENT, dims=2)

        # Get average embedding
        avg_embedding = model.backbone(source_imgs)
        avg_embedding = t.mean(avg_embedding, dim=0).unsqueeze(0)
        target_embedding = model.backbone(target_displaced.unsqueeze(0))

        histogram = model.match_corr(target_embedding, avg_embedding, padding=PAD)
        histogram = model.out_batchnorm(histogram)
        histogram = histogram.squeeze(1).squeeze(1)

        histogram = (histogram - t.mean(histogram)) / t.std(histogram)
        histogram = t.softmax(histogram, dim=1).squeeze(0).cpu()

        # visualize:
        interp_hist = interpolate_histogram(WIDTH, histogram)
        interp_hist_idx = np.argmax(interp_hist)
        predicted_shift = -(WIDTH//2 - interp_hist_idx)
        print("Estimated displacement is", predicted_shift, "pixels.")
        plot_displacement(target.squeeze(0).cpu(),
                          target.squeeze(0).cpu(),
                          histogram,
                          displacement=DISPLACEMENT,
                          name="result",
                          dir="./")


if __name__ == '__main__':
    get_average_embedding()
