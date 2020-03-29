import os
import numpy as np
import mxnet as mx

from tqdm import trange
from pathlib import Path
from datetime import datetime
from mxnet import gluon, autograd, nd

from models.mxnet_models import MultiDAE, MultiVAE
from utils.parser import parse_args
from utils.data_loader import DataLoader
from utils.metrics import NDCG_binary_at_k_batch, Recall_at_k_batch
from utils.reduce_lr_on_plateau import ReduceLROnPlateau

ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()


def early_stopping(curr_value, best_value, stop_step, patience, score_fn="loss"):
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

    running_loss, update_count = 0.0, 0
    N = data.shape[0]
    idxlist = list(range(N))
    np.random.shuffle(idxlist)
    training_steps = len(range(0, N, args.batch_size))

    with trange(train_steps) as t:
        for batch_idx, start_idx in zip(t, range(0, N, args.batch_size)):
            t.set_description("epoch: {}".format(epoch + 1))

            end_idx = min(start_idx + args.batch_size, N)
            X_inp = data[idxlist[start_idx:end_idx]]
            X_inp = nd.from_numpy(X_inp.toarray()).as_in_context(ctx)

            with autograd.record():
                if model.__class__.__name__ == "MultiVAE":
                    if args.total_anneal_steps > 0:
                        anneal = min(
                            args.anneal_cap, 1.0 * update_count / args.total_anneal_steps
                        )
                    else:
                        anneal = args.anneal_cap
                    update_count += 1
                    loss = model(X_inp, anneal)
                elif model.__class__.__name__ == "MultiDAE":
                    loss = model(X_inp)

            trainer.step(X_inp.shape[0])
            running_loss += nd.mean(loss).asscalar()
            avg_loss = running_loss / (batch_idx + 1)
            t.set_postfix(loss=avg_loss)


def eval_step(data_tr, data_te, data_type="valid"):

    running_loss, update_count = 0.0, 0
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

            X_tr_inp = nd.from_numpy(X_inp.toarray()).as_in_context(ctx)

            with autograd.predict_mode():
                if model.__class__.__name__ == "MultiVAE":
                    if args.total_anneal_steps > 0:
                        anneal = min(
                            args.anneal_cap, 1.0 * update_count / args.total_anneal_steps
                        )
                    else:
                    anneal = args.anneal_cap
                    loss = model(X_tr_inp, anneal)
                elif model.__class__.__name__ == "MultiDAE":
                    loss = models(X_tr_inp)

            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)

            # Exclude examples from training set
            X_out = X_out.asnumpy()
            X_out[X_tr.nonzero()] = -np.inf

            n100 = NDCG_binary_at_k_batch(X_out, X_te, k=100)
            r20 = Recall_at_k_batch(X_out, X_te, k=20)
            r50 = Recall_at_k_batch(X_out, X_te, k=50)
            n100_list.append(n100)
            r20_list.append(r20)
            r50_list.append(r20)

            t.set_postfix(loss=avg_loss)

        n100_list = np.concatenate(n100_list)
        r20_list = np.concatenate(r20_list)
        r50_list = np.concatenate(r50_list)

    return avg_loss, np.mean(n100_list), np.mean(r20_list), np.mean(r50_list)


if __name__ == "__main__":

    args = parse_args()
    DATA_DIR = Path("data")
    data_path = DATA_DIR / "_".join([args.dataset, "processed"])
    model_name = "_".join([args.model, str(datetime.now()).replace(" ", "_")])

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
        import pdb; pdb.set_trace()  # breakpoint 61385d7a //

    model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    model.hybridize()
    adam_optimizer = mx.optimizer.Adam(learning_rate=args.lr, wd=args.weight_decay)
    trainer = gluon.Trainer(model.collect_params(), optimizer=adam_optimizer)

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
    update_count = 0
    stop_step = 0
    for epoch in range(args.n_epochs):
        train_step(model, optimizer, train_data, epoch)
        if epoch % args.eval_every == (args.eval_every - 1):
            val_loss, n100, r20, r50 = eval_step(valid_data_tr, valid_data_te)
            if args.early_stop_score_fn == "loss":
                early_stop_score = val_loss
            elif args.early_stop_score_fn == "metric":
                early_stop_score = n100
            best_score, stop_step, stop = early_stopping(
                early_stop_score, best_score, stop_step, args.early_stop_patience
            )
            if args.lr_scheduler:
                scheduler.step(early_stop_score)
            print("=" * 80)
            print(
                "| valid loss {:4.2f} | n100 {:4.2f} | r20 {:4.2f} | "
                "r50 {:4.2f}".format(val_loss, n100, r20, r50)
            )
            print("=" * 80)
        if stop:
            break
        if (stop_step == 0) & (args.save_results):
            best_epoch = epoch
            model.save_parameters(str(model_weights / (model_name + ".params")))

    # Run on test data.
    test_loss, n100, r20, r50 = eval_step(test_data_tr, test_data_te, data_type="test")
    print("=" * 80)
    print(
        "| End of training | test loss {:4.2f} | n100 {:4.2f} | r20 {:4.2f} | "
        "r50 {:4.2f}".format(test_loss, n100, r20, r50)
    )
    print("=" * 80)

    # Save results
    if args.save_results:
        results_d = {}
        results_d["args"] = args.__dict__
        results_d["best_epoch"] = best_epoch
        results_d["loss"] = test_loss
        results_d["n100"] = n100
        results_d["r20"] = r20
        results_d["r50"] = r50
        pickle.dump(results_d, open((log_dir / model_name) + ".p", "wb"))
