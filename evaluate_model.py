from pathlib import Path

import numpy as np
import torch as t
from scipy import interpolate
from tqdm import tqdm

from configs import CONFIG
from model import get_parametrized_model

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

DATASET = "teach_and_repeat_2022_70cm"
VISUALIZE = False
PLOT_IMPORTANCES = False
WIDTH = 640  # - 8
CROP_SIZE = 248
PAD = 15
FRACTION = 8
OUTPUT_SIZE = WIDTH // FRACTION  #  WIDTH // FRACTION
CROPS_MULTIPLIER = 1
BATCHING = CROPS_MULTIPLIER    # this improves evaluation speed by a lot
MASK = t.zeros(OUTPUT_SIZE)
MASK[:PAD] = t.flip(t.arange(0, PAD), dims=[0])
MASK[-PAD:] = t.arange(0, PAD)
MASK = OUTPUT_SIZE - 1 - MASK
MASK = (OUTPUT_SIZE - 1) / MASK.to(device)
LAYER_POOL = False
FILTER_SIZE = 3
EMB_CHANNELS = 256
RESIDUALS = 0

EVAL_LIMIT = 1000
TOLERANCE = 32

MODEL_TYPE = "siam"
MODEL = "model_riseholme"

crops_num = int((WIDTH // CROP_SIZE) * CROPS_MULTIPLIER)
crops_idx = np.linspace(0, WIDTH-CROP_SIZE, crops_num, dtype=int) + FRACTION // 2

# crops_idx = np.array([WIDTH // 2 - CROP_SIZE // 2])
# crops_num = 1
histograms = np.zeros((1000, 64))


args = CONFIG("best_params")
NAME = args.nm
LR = 10**-args.lr
BATCH_SIZE = args.bs
NEGATIVE_FRAC = args.nf
# device = args.dev
LAYER_POOL = args.lp
FILTER_SIZE = args.fs
EMB_CHANNELS = args.ech
SMOOTHNESS = args.sm
assert args.res in [0, 1, 2, 3], "Residual type is wrong"
RESIDUALS = args.res


model = get_parametrized_model(LAYER_POOL, FILTER_SIZE, EMB_CHANNELS, RESIDUALS, PAD, device)


def get_histogram(src, tgt):
    target_crops = []
    for crop_idx in crops_idx:
        target_crops.append(tgt[..., crop_idx:crop_idx + CROP_SIZE])
    target_crops = t.cat(target_crops, dim=0)
    batched_source = src.repeat(crops_num // BATCHING, 1, 1, 1)
    # batched_source = t.zeros_like(batched_source)
    # batched_source = src
    histogram = model(batched_source, target_crops, padding=PAD)  # , fourrier=True)
    # histogram = histogram * MASK
    # histogram = t.sigmoid(histogram)
    # std, mean = t.std_mean(histogram, dim=-1, keepdim=True)
    # histogram = (histogram - mean) / std
    histogram = t.softmax(histogram, dim=1)
    return histogram


def get_importance(src, tgt, displac):
    # displac here is in size of embedding (OUTPUT_SIZE)
    histogram = model(src, tgt, padding=PAD, displac=displac).cpu().numpy()
    f = interpolate.interp1d(np.linspace(0, 1000, OUTPUT_SIZE), histogram, kind="cubic")
    interpolated = f(np.arange(512))
    return interpolated[0]


# def eval_displacement(epoch, eval_data=None, eval_model=None, dir="./results/eval_displacement/"):
#     def get_padding(output_shape):
#         return int((output_shape - 2) / 2)

#     global model
#     if eval_model is not None:
#         model = eval_model
#     else:
#         raise NotImplementedError

#     if eval_data is None:
#         raise NotImplementedError

#     pad = get_padding(WIDTH//FRACTION)

#     model.eval()
#     with torch.no_grad():
#         abs_err = 0
#         valid = 0
#         idx = 0
#         errors = []
#         results = []
#         for batch in tqdm(eval_data):
#             source, target, displ = batch[0].to(device), batch[1][..., FRACTION//2:-FRACTION//2].to(device), batch[2]

#             # do it in both directions target -> source and source -> target
#             histogram = model(source.unsqueeze(0), target.unsqueeze(0), padding=pad)
#             predicted_displ = np.argmax(histogram.cpu().numpy())
            
#             pixel_err = (predicted_displ - histogram.cpu().shape[-1] / 2) / (histogram.cpu().shape[-1] / 2) * WIDTH / 2
#             tmp_err = (pixel_err - displ)
#             abs_err += abs(tmp_err)
#             results.append(pixel_err)
#             errors.append(tmp_err)

#             if abs(pixel_err - displ) < TOLERANCE:
#                 valid += 1
#             idx += 1

#             if idx > EVAL_LIMIT:
#                 break

#         Path(f"{dir}").mkdir(parents=True, exist_ok=True)
#         print("Evaluated:", "\nAbsolute mean error:", abs_err/idx, "\nPredictions in tolerance:", valid*100/idx, "%")
#         np.savetxt(f"{dir}/eval_errors_{epoch}.csv", np.array(errors) * 2.0, delimiter=",")
#         # np.savetxt("results_" + MODEL_TYPE + "/eval_" + MODEL + "/" + DATASET + "_preds.csv", np.array(errors) * 2.0, delimiter=",")
#         # np.savetxt("results_" + MODEL_TYPE + "/eval_" + MODEL + "/" + DATASET + "_histograms.csv", histograms, delimiter=",")
#         return abs_err/idx, valid*100/idx


if __name__ == '__main__':
    pass
    #eval_displacement()
