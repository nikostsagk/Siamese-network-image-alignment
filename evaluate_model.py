import torch
import torch as t
from model import Siamese, load_model, get_custom_CNN, Transformer
from torch.utils.data import DataLoader
from parser_grief import ImgPairDataset, CroppedImgPairDataset
from torchvision.transforms import Resize
from tqdm import tqdm
from utils import get_shift, plot_samples, plot_displacement, affine
import numpy as np
from scipy import interpolate

VISUALIZE = False
WIDTH = 512
CROP_SIZE = 152
PAD = (CROP_SIZE - 8) // 16
FRACTION = 8
OUTPUT_SIZE = WIDTH // FRACTION
CROPS_MULTIPLIER = 32
BATCHING = CROPS_MULTIPLIER    # this improves evaluation speed by a lot


# transformer params
D_MODEL = 576
LAYERS = 4
HEADS = 8
DIM = 256

EVAL_LIMIT = 1000
TOLERANCE = 75

MODEL_TYPE = "siam"
MODEL = "cutout72_size16/model_32"
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
# device = t.device("cpu")

# backbone = get_pretrained_VGG11()   # use pretrained network - PAD = 7
backbone = get_custom_CNN()  # use custom network trained from scratch PAD = 3
if MODEL_TYPE == "siam":
    model = Siamese(backbone, padding=PAD).to(device)
elif MODEL_TYPE == "attn":
    model = Transformer(backbone, D_MODEL, LAYERS, HEADS, DIM).to(device).float()
model = load_model(model, "/home/zdeeno/Documents/Work/alignment/results_" + MODEL_TYPE + "/" + MODEL + ".pt")

transform = Resize(192)
# transform = Resize(192 * 2)
# transform = Resize((288, WIDTH))
crops_num = int(((WIDTH - CROP_SIZE) // CROP_SIZE) * CROPS_MULTIPLIER)
crops_idx = np.linspace(0, WIDTH-CROP_SIZE, crops_num, dtype=int)
print(crops_num, np.array(crops_idx))


def eval_displacement():
    dataset = ImgPairDataset(dataset="stromovka")
    train_loader = DataLoader(dataset, 1, shuffle=False)

    model.eval()
    with torch.no_grad():
        abs_err = 0
        valid = 0
        idx = 0
        errors = []
        for batch in tqdm(train_loader):
            source, target, displ = transform(batch[0].to(device)), transform(batch[1].to(device)), batch[2]

            def get_histogram(src, tgt):
                target_crops = []
                for crop_idx in crops_idx:
                    target_crops.append(tgt[..., crop_idx:crop_idx+CROP_SIZE])
                target_crops = t.cat(target_crops, dim=0)
                batched_source = src.repeat(crops_num//BATCHING, 1, 1, 1)
                # batched_source = src
                histogram = model(batched_source, target_crops)
                histogram = t.sigmoid(histogram)
                # histogram[:, 0] = 0  # filter boundary values
                # histogram[:, -1] = 0  # filter boundary values
                std, mean = t.std_mean(histogram, dim=-1, keepdim=True)
                histogram = (histogram - mean)/std
                return histogram

            # do it in both directions target -> source and source -> target
            histogram = get_histogram(source, target)
            shift_hist = get_shift(WIDTH, CROP_SIZE, histogram, crops_idx)
            histogram = get_histogram(target, source)
            shift_hist += t.flip(get_shift(WIDTH, CROP_SIZE, histogram, crops_idx), dims=(-1, ))

            f = interpolate.interp1d(np.linspace(0, 1024, OUTPUT_SIZE), shift_hist, kind="linear")
            interpolated = f(np.arange(1024))
            # interpolated = np.interp(np.arange(0, 1024), np.linspace(0, 1024, OUTPUT_SIZE), shift_hist.numpy())
            ret = -(np.argmax(interpolated) - 512)
            displac_mult = 1024/WIDTH
            abs_err += abs(ret - displ.numpy()[0])/displac_mult
            errors.append((ret - displ.numpy()[0])/displac_mult)
            if VISUALIZE and abs(ret - displ.numpy()[0]) >= TOLERANCE:
                plot_displacement(source.squeeze(0).cpu(),
                                  target.squeeze(0).cpu(),
                                  shift_hist.squeeze(0).cpu(),
                                  displacement=-displ.numpy()[0]/displac_mult,
                                  name=str(idx),
                                  dir="results_" + MODEL_TYPE + "/eval_" + MODEL + "/")
                print(ret, displ.numpy()[0])

            idx += 1
            if abs(ret - displ.numpy()[0]) < TOLERANCE:
                valid += 1

            if idx > EVAL_LIMIT:
                break

        print("Evaluated:", "\nAbsolute mean error:", abs_err/idx, "\nPredictions in tolerance:", valid*100/idx, "%")
        np.savetxt("results_" + MODEL_TYPE + "/eval_" + MODEL + "/errors.csv", errors, delimiter=",")


def eval_heatmap():
    dataset = CroppedImgPairDataset(CROP_SIZE, FRACTION, 0, transforms=transform)
    train_loader = DataLoader(dataset, 1, shuffle=False)

    model.eval()
    with torch.no_grad():
        idx = 0
        for batch in tqdm(train_loader):
            source, target, heatmap = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            histogram = model(source, target)
            histogram = t.sigmoid(histogram)
            histogram[:, 0] = 0  # filter boundary values
            histogram[:, -1] = 0  # filter boundary values

            plot_samples(source.squeeze(0).cpu(),
                         target.squeeze(0).cpu(),
                         heatmap.squeeze(0).cpu(),
                         prediction=histogram.squeeze(0).cpu(),
                         name=str(idx),
                         dir="results_" + MODEL_TYPE + "/eval_" + MODEL + "/")

            idx += 1


if __name__ == '__main__':
    eval_displacement()
    # eval_heatmap()