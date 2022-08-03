#!/usr/bin/env python3.9
import argparse
import copy
from pathlib import Path

import numpy as np
import torch as t
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate_model import TOLERANCE
from model import get_parametrized_model, load_model, save_model
from parser_polytunnel import RiseholmePolytunnel
from utils import batch_augmentations, interpolate_histogram, plot_displacement2, plot_samples

#import wandb

def get_pad(crop, fraction):
    return (crop - fraction) // (2 * fraction)

VISUALISE = True
WANDB = False
NAME = "siam_finetuned"
BATCH_SIZE = 32  # higher better
EPOCHS = 128
LR = 5
EVAL_RATE = 1
CROP_SIZES = 248
FRACTION = 8
PAD = get_pad(CROP_SIZES, FRACTION)
SMOOTHNESS = 3
OUTPUT_SIZE = 64
TOLERANCE = 32

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
batch_augmentations = batch_augmentations.to(device)

parser = argparse.ArgumentParser(description='Training script for image alignment.')
parser.add_argument('--nm', type=str, help="name of the model", default=NAME)
parser.add_argument('--lr', type=float, help="learning rate", default=LR)
parser.add_argument('--bs', type=int, help="batch size", default=BATCH_SIZE)
# parser.add_argument('--dev', type=str, help="device", required=True)
parser.add_argument('--sm', type=int, help="smoothness of target", default=SMOOTHNESS)
parser.add_argument('--cs', type=int, help="crop size", default=CROP_SIZES)
args = parser.parse_args()

print("Argument values: \n", args)
NAME = args.nm
LR = 10**-args.lr
BATCH_SIZE = args.bs
CROP_SIZES = args.cs
SMOOTHNESS = args.sm
DATA_PATH = "/home/nikos/shared/data_collection/teach_and_repeat_2022_70cm"
EVAL_PATH = "/home/nikos/shared/results_v2"

dataset = RiseholmePolytunnel(CROP_SIZES, FRACTION, SMOOTHNESS, path=DATA_PATH)
val, train = t.utils.data.random_split(dataset, [int(0.05 * len(dataset)), int(0.95 * len(dataset)) + 1])

train_loader = DataLoader(train, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val, BATCH_SIZE, shuffle=False)
eval_data = RiseholmePolytunnel(CROP_SIZES, FRACTION, SMOOTHNESS, eval=True)

model = get_parametrized_model(False, 3, 256, 0, PAD, device)
optimizer = AdamW(model.parameters(), lr=LR)
loss = BCEWithLogitsLoss()
in_example = (t.zeros((1, 3, 384, 512)).to(device).float(), t.zeros((1, 3, 384, 512)).to(device).float())


def train_loop(epoch):
    PAD = get_pad(CROP_SIZES, FRACTION)
    model.train()
    loss_sum = 0
    print("Training model epoch", epoch)
    for batch in tqdm(train_loader):
        source, target, heatmap = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        source = batch_augmentations(source)

        out = model(source, target, padding=PAD)
        optimizer.zero_grad()

        l = loss(out, heatmap)
        loss_sum += l.cpu().detach().numpy()
        l.backward()
        optimizer.step()

    # if epoch % EVAL_RATE == 0 and WANDB:
    #     wandb.log({"epoch": epoch, "train_loss": loss_sum / len(train_loader)})
    print("Training of epoch", epoch, "ended with loss", loss_sum / len(train_loader))


def eval_loop(epoch):
    PAD = get_pad(CROP_SIZES, FRACTION)
    model.eval()
    with t.no_grad():
        print("Validating model after epoch", epoch)
        loss_sum = 0
        idx = 0
        for batch in tqdm(val_loader):
            source, target, heatmap = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            source = batch_augmentations(source)

            out = model(source, target, padding=PAD)
            l = loss(out, heatmap)
            loss_sum += l.cpu().detach().numpy()
            idx += 1

            if VISUALISE:
                vis_out = (out - t.mean(out, dim=1).unsqueeze(1)) / t.std(out, dim=1).unsqueeze(1)
                vis_out = t.softmax(vis_out, dim=1)
                plot_samples(source[0].cpu(),
                             target[0].cpu(),
                             heatmap[0].cpu(),
                             prediction=vis_out[0].cpu(),
                             name=f"batch_{idx}",
                             dir=f"{EVAL_PATH}/plot_samples/epoch_{str(epoch)}/", save=True)

        print("Evaluating for displacement:")
        mae, acc = eval_displacement(epoch=epoch, eval_data=eval_data, eval_model=model, dir=f"{EVAL_PATH}/eval_displacement/")
        # if WANDB:
        #     wandb.log({"val_loss": loss_sum/len(val_loader), "MAE": mae, "Accuracy": acc})
        #     wandb.watch(model)
        print(f"Epoch {epoch} validation loss {loss_sum/len(val_loader)} MAE {mae} Accuracy {acc}")

    return mae

def eval_displacement(epoch, eval_data=None, eval_model=None, dir="./results/eval_displacement/"):
    def get_padding(output_shape):
        return int((output_shape - 2) / 2)

    model = eval_model
    pad = get_padding(OUTPUT_SIZE)

    model.eval()
    with t.no_grad():
        abs_err = 0
        valid = 0
        idx = 0
        errors = []
        results = []
        for batch in tqdm(eval_data):
            source, target, displ = batch[0].to(device), batch[1][..., FRACTION//2:-FRACTION//2].to(device), batch[2]

            # do it in both directions target -> source and source -> target
            histogram = model(source.unsqueeze(0), target.unsqueeze(0), padding=pad)
            histogram = (histogram - t.mean(histogram)) / t.std(histogram)
            histogram = t.softmax(histogram, dim=1).squeeze(0).cpu()
            interp_hist = interpolate_histogram(source.shape[-1], histogram.squeeze(0))
            interp_hist_idx = np.argmax(interp_hist)
            
            pixel_err = -(source.shape[-1]//2 - interp_hist_idx)
            tmp_err = (pixel_err - displ)
            abs_err += abs(tmp_err)
            results.append(pixel_err)
            errors.append(tmp_err)

            if abs(pixel_err - displ) < TOLERANCE:
                valid += 1
            idx += 1

            if idx > 1000:
                break

        Path(f"{dir}").mkdir(parents=True, exist_ok=True)
        print("Evaluated:", "\nAbsolute mean error:", abs_err/idx, "\nPredictions in tolerance:", valid*100/idx, "%")
        np.savetxt(f"{dir}/eval_errors_{epoch}.csv", np.array(errors), delimiter=",")
        return abs_err/idx, valid*100/idx


LOAD_EPOCH = 0
# model, optimizer = load_model(model, "/home/nikos/Siamese-network-image-alignment/model_siam.pt", optimizer=optimizer)

# if WANDB:
#     wandb.init(project="alignment", entity="zdeeno", config=vars(args))

lowest_err = 1000
best_model = None

for epoch in range(LOAD_EPOCH, EPOCHS):
    if epoch % EVAL_RATE == 0:
        err = eval_loop(epoch)
        if err < lowest_err:
            lowest_err = err
            best_model = copy.deepcopy(model)
            save_model(model, NAME, epoch, optimizer, dir=f"{EVAL_PATH}/models/")

    train_loop(epoch)
