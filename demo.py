import numpy as np
import torch as t
from torchvision.io import read_image
from torchvision.transforms import Resize

from average_embedding import FRACTION, IMAGE_WIDTH
from model import get_parametrized_model, load_model
from utils import interpolate_histogram, plot_displacement

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

#MODEL_PATH = "./model_siam.pt"
MODEL_PATH = "/home/nikos/shared/results_v1/models/siam_109.pt"
IMG1_PATH = "../shared/data_collection/teach_and_repeat_2022_70cm/62da663663002c794c29b6cd/t0-r2-c0_S.png"
IMG2_PATH = "../shared/data_collection/teach_and_repeat_2022_70cm/62da663663002c794c29b6cd/t1-r1-c0_S.png"
# -------------------------------


WIDTH = 512
OUTPUT_SIZE = 64
FRACTION = int(WIDTH / OUTPUT_SIZE)
PAD = int((OUTPUT_SIZE - 2) / 2)


def run_demo():
    # Read images (they should be of the same shape)
    source, target = read_image(IMG1_PATH) / 255.0, read_image(IMG2_PATH) / 255.0
    transform = Resize(int(source.shape[1] * WIDTH / source.shape[2]))
    source, target = transform(source).to(device), transform(target).to(device)[..., FRACTION//2:-FRACTION//2]

    model = get_parametrized_model(False, 3, 256, 0, PAD, device)
    model = load_model(model, MODEL_PATH)

    model.eval()
    with t.no_grad():
        print(f"Source img: {source.shape}, Target img: {target.shape}")

        histogram = model(source.unsqueeze(0), target.unsqueeze(0), padding=PAD)
        histogram = (histogram - t.mean(histogram)) / t.std(histogram)
        histogram = t.softmax(histogram, dim=1).squeeze(0).cpu()

        # visualize:
        interp_hist = interpolate_histogram(WIDTH, histogram)
        interp_hist_idx = np.argmax(interp_hist)
        predicted_shift = -(WIDTH//2 - interp_hist_idx)

        print("Estimated displacement is", predicted_shift, "pixels.")
        plot_displacement(source.squeeze(0).cpu(),
                          target.squeeze(0).cpu(),
                          histogram,
                          displacement=None,
                          name="result",
                          dir="./")

if __name__ == '__main__':
    run_demo()
