import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import trange
from pathlib import Path
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.pytorch_models import MultiDAE, MultiVAE
from utils.parser import parse_args
from utils.data_loader import DataLoader
from utils.metrics import NDCG_binary_at_k_batch, Recall_at_k_batch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def init_weights(model):
    for name, param in model.named_parameters():
        if "weight" in name:
            nn.init.xavier_uniform_(param.data)
        elif "bias" in name:
            param.data.normal_(std=0.001)


def vae_loss_fn(inp, out, mu, logvar, anneal):
    neg_ll = -torch.mean(torch.sum(F.log_softmax(out, 1) * inp, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return neg_ll + anneal * KLD


def early_stopping(curr_value, best_value, stop_step, patience, score_fn):
    if (score_fn == "loss" and curr_value <= best_value) or (
        score_fn == "metric" and curr_value >= best_value
    ):
        stop_step, best_value = 0, curr_value
    else:
        stop_step += 1
    if stop_step >= patience:
        print(
            "Early stopping triggered. patience: {} log:{}".format(patience, best_value)
        )
        stop = True
    else:
        stop = False
    return best_value, stop_step, stop


def train_step(model, optimizer, data, epoch):

    model.train()
    running_loss = 0.0
    global update_count
    N = data.shape[0]
    idxlist = list(range(N))
    np.random.shuffle(idxlist)
    training_steps = len(range(0, N, args.batch_size))

    with trange(training_steps) as t:
        for batch_idx, start_idx in zip(t, range(0, N, args.batch_size)):
            t.set_description("epoch: {}".format(epoch + 1))

            end_idx = min(start_idx + args.batch_size, N)
            X_inp = data[idxlist[start_idx:end_idx]]
            X_inp = torch.FloatTensor(X_inp.toarray()).to(device)

            if args.constant_anneal:
                anneal = args.anneal_cap
            else:
                anneal = min(args.anneal_cap, update_count / total_anneal_steps)
            update_count += 1

            optimizer.zero_grad()
            if model.__class__.__name__ == "MultiVAE":
                X_out, mu, logvar = model(X_inp)
                loss = vae_loss_fn(X_inp, X_out, mu, logvar, anneal)
                train_step.anneal = anneal
            elif model.__class__.__name__ == "MultiDAE":
                X_out = model(X_inp)
                loss = -torch.mean(torch.sum(F.log_softmax(X_out, 1) * X_inp, -1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)

            t.set_postfix(loss=avg_loss)


def eval_step(data_tr, data_te, data_type="valid"):

    model.eval()
    running_loss = 0.0
    eval_idxlist = list(range(data_tr.shape[0]))
    eval_N = data_tr.shape[0]
    eval_steps = len(range(0, eval_N, args.batch_size))

    n100_list, r20_list, r50_list = [], [], []

    with trange(eval_steps) as t:
        with torch.no_grad():
            for batch_idx, start_idx in zip(t, range(0, eval_N, args.batch_size)):
                t.set_description(data_type)

                end_idx = min(start_idx + args.batch_size, eval_N)
                X_tr = data_tr[eval_idxlist[start_idx:end_idx]]
                X_te = data_te[eval_idxlist[start_idx:end_idx]]
                X_tr_inp = torch.FloatTensor(X_tr.toarray()).to(device)

                if model.__class__.__name__ == "MultiVAE":
                    X_out, mu, logvar = model(X_tr_inp)
                    loss = vae_loss_fn(X_tr_inp, X_out, mu, logvar, train_step.anneal)
                elif model.__class__.__name__ == "MultiDAE":
                    X_out = model(X_tr_inp)
                    loss = -torch.mean(
                        torch.sum(F.log_softmax(X_out, 1) * X_tr_inp, -1)
                    )
                running_loss += loss.item()
                avg_loss = running_loss / (batch_idx + 1)

                # Exclude examples from training set
                X_out = X_out.cpu().numpy()
                X_out[X_tr.nonzero()] = -np.inf

                n100 = NDCG_binary_at_k_batch(X_out, X_te, k=100)
                r20 = Recall_at_k_batch(X_out, X_te, k=20)
                r50 = Recall_at_k_batch(X_out, X_te, k=50)
                n100_list.append(n100)
                r20_list.append(r20)
                r50_list.append(r50)

                t.set_postfix(loss=avg_loss)

        n100_list = np.concatenate(n100_list)
        r20_list = np.concatenate(r20_list)
        r50_list = np.concatenate(r50_list)

    return avg_loss, np.mean(n100_list), np.mean(r20_list), np.mean(r50_list)


if __name__ == "__main__":

    args = parse_args()
    DATA_DIR = Path("data")
    data_path = DATA_DIR / "_".join([args.dataset, "processed"])
    model_name = "_".join(["pt", args.model, str(datetime.now()).replace(" ", "_")])

    log_dir = Path(args.log_dir)
    model_weights = log_dir / "weights"
    if not os.path.exists(model_weights):
        os.makedirs(model_weights)

    data_loader = DataLoader(data_path)
    n_items = data_loader.n_items
    train_data = data_loader.load_data("train")
    valid_data_tr, valid_data_te = data_loader.load_data("validation")
    test_data_tr, test_data_te = data_loader.load_data("test")

    training_steps = len(range(0, train_data.shape[0], args.batch_size))
    try:
        total_anneal_steps = (
            training_steps * (args.n_epochs - int(args.n_epochs * 0.15))
        ) / args.anneal_cap
    except ZeroDivisionError:
        assert (
            args.constant_anneal
        ), "if 'anneal_cap' is set to 0.0 'constant_anneal' must be set to 'True"

    p_dims = eval(args.p_dims)
    q_dims = eval(args.q_dims)
    if q_dims:
        assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
        assert (
            q_dims[-1] == p_dims[0]
        ), "Latent dimension for p- and q- network mismatches."
    else:
        q_dims = p_dims[::-1]
    q_dims = [n_items] + q_dims
    p_dims = p_dims + [n_items]
    dropout_enc = eval(args.dropout_enc)
    dropout_dec = eval(args.dropout_dec)

    if args.model == "vae":
        model = MultiVAE(
            p_dims=p_dims,
            q_dims=q_dims,
            dropout_enc=dropout_enc,
            dropout_dec=dropout_dec,
        )
    elif args.model == "dae":
        model = MultiDAE(
            p_dims=p_dims,
            q_dims=q_dims,
            dropout_enc=dropout_enc,
            dropout_dec=dropout_dec,
        )

    init_weights(model)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.lr_scheduler:
        if args.early_stop_score_fn == "loss":
            scheduler = ReduceLROnPlateau(
                optimizer,
                patience=args.lr_patience,
                factor=0.4,
                threshold=2,
                threshold_mode="abs",
            )
        if args.early_stop_score_fn == "metric":
            scheduler = ReduceLROnPlateau(
                optimizer,
                patience=args.lr_patience,
                factor=0.4,
                threshold=0.001,
                threshold_mode="rel",
            )

    if args.early_stop_score_fn == "loss":
        best_score = np.inf
    elif args.early_stop_score_fn == "metric":
        best_score = -np.inf
    stop_step = 0
    update_count = 0
    stop = False
    for epoch in range(args.n_epochs):
        train_step(model, optimizer, train_data, epoch)
        if epoch % args.eval_every == (args.eval_every - 1):
            val_loss, n100, r20, r50 = eval_step(valid_data_tr, valid_data_te)
            if args.early_stop_score_fn == "loss":
                early_stop_score = val_loss
            elif args.early_stop_score_fn == "metric":
                early_stop_score = n100
            best_score, stop_step, stop = early_stopping(
                early_stop_score,
                best_score,
                stop_step,
                args.early_stop_patience,
                args.early_stop_score_fn,
            )
            if args.lr_scheduler:
                scheduler.step(early_stop_score)
            print("=" * 80)
            print(
                "| valid loss {:4.3f} | n100 {:4.3f} | r20 {:4.3f} | "
                "r50 {:4.3f}".format(val_loss, n100, r20, r50)
            )
            print("=" * 80)
        if stop:
            break
        if (stop_step == 0) & (args.save_results):
            best_epoch = epoch
            torch.save(model.state_dict(), model_weights / (model_name + ".pt"))

    if args.save_results:
        # Run on test data with best model
        model.load_state_dict(torch.load(model_weights / (model_name + ".pt")))
        test_loss, n100, r20, r50 = eval_step(
            test_data_tr, test_data_te, data_type="test"
        )
        print("=" * 80)
        print(
            "| End of training | test loss {:4.3f} | n100 {:4.3f} | r20 {:4.3f} | "
            "r50 {:4.3f}".format(test_loss, n100, r20, r50)
        )
        print("=" * 80)

        # Save results
        results_d = {}
        results_d["args"] = args.__dict__
        results_d["best_epoch"] = best_epoch
        results_d["loss"] = test_loss
        results_d["n100"] = n100
        results_d["r20"] = r20
        results_d["r50"] = r50
        pickle.dump(results_d, open(str(log_dir / (model_name + ".p")), "wb"))
