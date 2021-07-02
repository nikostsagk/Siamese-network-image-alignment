import random
import torch as t
from parser_nordland import CroppedImgPairDataset
from model import Siamese, get_custom_CNN, get_pretrained_VGG11, save_model, load_model
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.functional import softmax
from tqdm import tqdm
from utils import plot_samples


BATCH_SIZE = 16
EPOCHS = 1000
LR = 1e-5
EVAL_RATE = 1
CROP_SIZE = 32
FRACTION = 16
PAD = 0
SMOOTHNESS = 0
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
# device = t.device("cpu")


dataset = CroppedImgPairDataset(CROP_SIZE, FRACTION, SMOOTHNESS, target_pad=True)
val, train = t.utils.data.random_split(dataset, [int(0.1 * len(dataset)), int(0.9 * len(dataset))])
train_loader = DataLoader(train, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val, 1, shuffle=False)

# backbone = get_pretrained_VGG11()   # use pretrained network - PAD = 7
backbone = get_custom_CNN()  # use custom network trained from scratch PAD = 3
model = Siamese(backbone, padding=PAD).to(device)
optimizer = AdamW(model.parameters(), lr=LR)
loss = CrossEntropyLoss()
# loss = BCEWithLogitsLoss()


def train_loop(epoch):
    model.train()
    loss_sum = 0
    print("Training model epoch", epoch)
    for batch in tqdm(train_loader):
        source, target, heatmap = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        out = model(source, target)
        # print(out.size(), heatmap.size())
        optimizer.zero_grad()
        # print(t.where(heatmap == 1)[1].view(BATCH_SIZE, 3))
        l = loss(out, t.argmax(heatmap.long(), dim=-1))
        # heatmap[heatmap > 0] = 1.0
        # l = loss(out, heatmap)
        loss_sum += l.cpu().detach().numpy()
        l.backward()
        optimizer.step()
    print("Training of epoch", epoch, "ended with loss", loss_sum / len(train_loader))


def eval_loop(epoch):
    model.eval()
    with t.no_grad():
        print("Evaluating model after epoch", epoch)
        for idx, batch in enumerate(val_loader):
            if idx % 10 == 0:
                source, target, heatmap = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                out = model(source, target)
                out = softmax(t.sigmoid(out.squeeze(0).cpu()), dim=0)
                plot_samples(source.squeeze(0).cpu(),
                             target.squeeze(0).cpu(),
                             heatmap.squeeze(0).cpu(),
                             prediction=out,
                             name=str(idx),
                             dir="results_siam/" + str(epoch) + "/")
                if idx > 100:
                    break


if __name__ == '__main__':
    # model, optimizer, load_model(model, "/home/zdeeno/Documents/Work/alignment/results/model_5.pt", optimizer=optimizer)

    for epoch in range(0, EPOCHS):
        save_model(model, "siam", epoch, optimizer)
        if epoch % EVAL_RATE == 0:
            eval_loop(epoch)
        train_loop(epoch)
