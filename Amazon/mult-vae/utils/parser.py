import argparse


def parse_args():

    parser = argparse.ArgumentParser(
        description="Variational Autoencoders for Collaborative Filtering"
    )

    parser.add_argument(
        "--dataset", type=str, default="amazon", help="one of: amazon or movielens"
    )
    parser.add_argument(
        "--log_dir", type=str, default="results", help="Store model path."
    )

    # Set model
    parser.add_argument(
        "--model", type=str, default="vae", help="Specify the model {vae, dae}."
    )

    # model parameters
    parser.add_argument(
        "--q_dims",
        type=str,
        default="None",
        help="encoder layer dimensions. These do not include the first layer of dim n_items",
    )
    parser.add_argument(
        "--p_dims",
        type=str,
        default="[200, 600]",
        help="decoder layer dimensions. These do not include the last layer of dim n_items",
    )
    parser.add_argument(
        "--dropout_enc", type=str, default="[0.5, 0.]", help="encoder dropout"
    )
    parser.add_argument(
        "--dropout_dec", type=str, default="[0., 0.]", help="decoder dropout"
    )

    # Train/Test parameters
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of epoch.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="l2 reg.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size.")
    parser.add_argument(
        "--total_anneal_steps",
        type=int,
        default=200000,
        help="total number of gradient updates for annealing",
    )
    parser.add_argument(
        "--anneal_cap", type=float, default=0.2, help="largest annealing parameter"
    )
    parser.add_argument(
        "--lr_scheduler", action="store_true", help="if true use ReduceLROnPlateau",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=3,
        help="Patience for ReduceLROnPlateau lr_scheduler before decreasing lr",
    )
    parser.add_argument(
        "--eval_every", type=int, default=1, help="Evaluate every N epochs"
    )
    parser.add_argument(
        "--early_stop_patience", type=int, default=2, help="Patience for early stopping"
    )
    parser.add_argument(
        "--early_stop_score_fn", type=str, default="loss", help="{loss, metric}"
    )

    # save parameters
    parser.add_argument(
        "--save_results", action="store_true", help="Save model and results"
    )

    return parser.parse_args()
