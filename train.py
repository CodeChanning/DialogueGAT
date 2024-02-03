from argparse import ArgumentParser
#argparse docs: https://docs.python.org/3/library/argparse.html

import torch
from torch import nn, optim

from gat import DialogueGAT
from utils import seed_everything, EarlyStopping


def cli_main():
    #sets seed for pseudo-random number generators
    #https://pytorch-lightning.readthedocs.io/en/1.7.7/api/pytorch_lightning.utilities.seed.html#pytorch_lightning.utilities.seed.seed_everything
    seed_everything(1234)

    # ------------
    # args
    # ------------

    #adding diff argument flags for parser cli
    parser = ArgumentParser()

    parser.add_argument("--data_path", type=str, default="data_sample/")
    parser.add_argument("--year", type=int, default=2015)
    parser.add_argument("--target", type=int, default=3)

    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--l2", type=float, default=1e-6)

    parser.add_argument("--use_gpu", action="store_true")

    #storing set args in parser to pass to GAT later
    parser = DialogueGAT.add_model_specific_args(parser)

    args = parser.parse_args()
    

    # ------------
    # model
    # ------------

    #TODO: not familiar with this pythong syntax
    (
        train_loader,
        val_loader,
        test_loader,
        model,
        optimizer,
        loss_func,
        device,
    ) = DialogueGAT.build_model(args)

    # ------------
    # training
    # ------------

    #early stoppping is used to stop training when no improvement is see after a given number of events.
    best_metrics, best_model, test_metrics = None, None, None
    early_stopping = EarlyStopping(
        "save/", patience=args.patience, verbose=True, mode="min"
    )

    #going through epochs specified by args in cli
    for e in range(args.epochs):

        #running epochs using gat
        print("\nEpoch-{}:".format(e + 1))
        train_metrics = model.run_epoch(train_loader, loss_func, optimizer, device)
        val_metrics = model.run_epoch(val_loader, loss_func, device=device, stage="Val")

        #TODO: idk what this check is for
        if not best_metrics or best_metrics["loss"] > val_metrics["loss"]:
            best_metrics = val_metrics
            test_metrics = test_metrics = model.run_epoch(
                test_loader, loss_func, device=device, stage="Test"
            )
            # torch.save(
            #     model.state_dict(),
            #     "../save/gat_{}_{}.pt".format(args.year, args.target),
            # )

        val_loss = val_metrics["loss"]
        print("train_loss: {loss:.4f}".format(**train_metrics))
        print("val_loss  : {loss:.4f}".format(**val_metrics))

        #early stopping if necessary
        early_stopping(val_metrics["loss"], model)
        if early_stopping.early_stop:
            print("Early stopping\n")
            break

    print("test_loss : {loss:.4f}".format(**test_metrics))


if __name__ == "__main__":
    cli_main()
