import pickle
import os
from datetime import datetime
from pathlib import Path

import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, nd
from tqdm import trange

from models.mxnet_models import MultiDAE, MultiVAE
from utils.data_loader import DataLoader
from utils.metrics import NDCG_binary_at_k_batch, Recall_at_k_batch
from utils.parser import parse_args
from utils.reduce_lr_on_plateau import ReduceLROnPlateau

ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()


def vae_loss_fn(inp, out, mu, logvar, anneal):
    neg_ll = -nd.mean(nd.sum(nd.log_softmax(out) * inp, -1))
    KLD = -0.5 * nd.mean(nd.sum(1 + logvar - nd.power(mu, 2) - nd.exp(logvar), axis=1))
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
            X_inp = nd.array(X_inp.toarray()).as_in_context(ctx)

            if args.constant_anneal:
                anneal = args.anneal_cap
            else:
                anneal = min(args.anneal_cap, update_count / total_anneal_steps)
            update_count += 1

            with autograd.record():
                if model.__class__.__name__ == "MultiVAE":
                    X_out, mu, logvar = model(X_inp)
                    loss = vae_loss_fn(X_inp, X_out, mu, logvar, anneal)
                    train_step.anneal = anneal
                elif model.__class__.__name__ == "MultiDAE":
                    X_out = model(X_inp)
                    loss = -nd.mean(nd.sum(nd.log_softmax(X_out) * X_inp, -1))
            loss.backward()
            trainer.step(X_inp.shape[0])
            running_loss += loss.asscalar()
            avg_loss = running_loss / (batch_idx + 1)

            t.set_postfix(loss=avg_loss)


def eval_step(data_tr, data_te, data_type="valid"):

    running_loss = 0.0
    eval_idxlist = list(range(data_tr.shape[0]))
    eval_N = data_tr.shape[0]
    eval_steps = len(range(0, eval_N, args.batch_size))

    n100_list, r20_list, r50_list = [], [], []

    with trange(eval_steps) as t:
        for batch_idx, start_idx in zip(t, range(0, eval_N, args.batch_size)):
            t.set_description(data_type)

            end_idx = min(start_idx + args.batch_size, eval_N)
            X_tr = data_tr[eval_idxlist[start_idx:end_idx]]
            X_te = data_te[eval_idxlist[start_idx:end_idx]]
            X_tr_inp = nd.array(X_tr.toarray()).as_in_context(ctx)

            with autograd.predict_mode():
                if model.__class__.__name__ == "MultiVAE":
                    X_out, mu, logvar = model(X_tr_inp)
                    loss = vae_loss_fn(X_tr_inp, X_out, mu, logvar, train_step.anneal)
                elif model.__class__.__name__ == "MultiDAE":
                    X_out = model(X_tr_inp)
                    loss = -nd.mean(nd.sum(nd.log_softmax(X_out) * X_tr_inp, -1))

            running_loss += loss.asscalar()
            avg_loss = running_loss / (batch_idx + 1)

            # Exclude examples from training set
            X_out = X_out.asnumpy()
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
    model_name = "_".join(["mx", args.model, str(datetime.now()).replace(" ", "_")])

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

    model.initialize(mx.init.Xavier(), ctx=ctx)
    model.hybridize()
    optimizer = mx.optimizer.Adam(learning_rate=args.lr, wd=args.weight_decay)
    trainer = gluon.Trainer(model.collect_params(), optimizer=optimizer)

    if args.lr_scheduler:
        if args.early_stop_score_fn == "loss":
            scheduler = ReduceLROnPlateau(
                trainer,
                patience=args.lr_patience,
                factor=0.4,
                threshold=2,
                threshold_mode="abs",
            )
        if args.early_stop_score_fn == "metric":
            scheduler = ReduceLROnPlateau(
                trainer,
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
            model.save_parameters(str(model_weights / (model_name + ".params")))

    if args.save_results:
        # Run on test data with best model
        model.load_parameters(str(model_weights / (model_name + ".params")), ctx=ctx)
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
