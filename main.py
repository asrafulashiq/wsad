from __future__ import print_function
import os
import torch
import torch.optim as optim

from model import Model
import options
import numpy as np
from test import test
from train import train
from dataset import Dataset
from tensorboard_logger import Logger
from tqdm import tqdm

torch.set_default_tensor_type("torch.cuda.FloatTensor")

if __name__ == "__main__":
    args = options.parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    print(args)

    dataset = Dataset(args, mode='both')
    if not os.path.exists("./ckpt/"):
        os.makedirs("./ckpt/")
    if not os.path.exists("./logs/" + args.model_name):
        os.makedirs("./logs/" + args.model_name)
    logger = Logger("./logs/" + args.model_name)

    model = Model(dataset.feature_size, dataset.num_class)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    init_itr = 0
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=10,
        verbose=True,
        threshold=0.1,
        min_lr=1e-8,
    )

    if args.pretrained_ckpt is not None:
        checkpoint = torch.load(args.pretrained_ckpt)
        model.load_state_dict(checkpoint["model_state_dict"])
        init_itr = checkpoint["itr"]

    if args.test:
        test(init_itr, dataset, args, model, logger, device)
        raise SystemExit

    best_dmap_itr = [0, init_itr]
    list_loss = []

    for itr in (range(init_itr, args.max_iter)):
        _loss = train(
            itr, dataset, args, model, optimizer, logger, device,
            scheduler=lr_scheduler
        )
        # list_loss.append(_loss)
        if itr % 100 == 0 and not itr == 0:
            model_state = model.state_dict()
            torch.save(
                {
                    "itr": itr,
                    "model_state_dict": model_state
                },
                "./ckpt/" + args.model_name + ".pkl",
            )

            # lr_scheduler.step(np.mean(list_loss))
            # list_loss = []

        if itr % 100 == 0 and not itr == 0:
            print("Iter: {}".format(itr))
            dmap = test(itr, dataset, args, model, logger, device)

            if dmap > best_dmap_itr[0]:
                torch.save(
                    {
                        "itr": itr,
                        "model_state_dict": model_state
                    },
                    "./ckpt/" + args.model_name + "_best" + ".pkl",
                )
                best_dmap_itr = [dmap, itr]
