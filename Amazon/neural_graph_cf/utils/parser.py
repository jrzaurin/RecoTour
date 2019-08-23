import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run NGCF.")
    parser.add_argument('--data_path', type=str, default='Data',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='gowalla',
                        help='Dataset name')
    parser.add_argument('--model_path', type=str, default='model_weights',
                        help='Store model path.')
    parser.add_argument('--results_path', type=str, default='results',
                        help='Store results path.')

    parser.add_argument('--adj_type', nargs='?', default='norm',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epoch.')
    parser.add_argument('--reg', type=float, default=1e-5,
                        help='l2 reg.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')

    parser.add_argument('--emb_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layers', type=str, default='[64,64]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--node_dropout', type=float, default=0.,
                        help='Graph Node dropout.')
    parser.add_argument('--edge_dropout', type=float, default=0.,
                        help='Graph edge dropout.')
    parser.add_argument('--mess_dropout', type=int, default=0.1,
                        help='Message dropout.')

    parser.add_argument('--n_folds', type=int, default=100,
                        help='number of partitions for the adjacency matrix')
    parser.add_argument('--Ks', type=str, default='[20, 40, 60, 80, 100]',
                        help='k order of metric evaluation (e.g. NDCG@k)')
    parser.add_argument('--test_with', type=str, default='cpu',
                        help='whether to use the GPU enabled test function')
    parser.add_argument('--print_every', type=int, default=1,
                        help='print results every N epochs')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Evaluate every N epochs')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save model every N epochs')

    return parser.parse_args()
