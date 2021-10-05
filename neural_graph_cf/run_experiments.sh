# Embeddings
python run.py --emb_dim 16 --layers [16,16]
python run.py --emb_dim 32 --layers [32,32]
python run.py --emb_dim 64 --layers [64,64]

# with lr schedulers starting with a higher learning rate
python run.py --emb_dim 64 --layers [64,64] --lr 0.003 --lr_scheduler ReduceLROnPlateau
python run.py --emb_dim 64 --layers [64,64] --lr 0.01 --lr_scheduler CyclicLR

# RAdam
python run.py --emb_dim 64 --layers [64,64] --lr 0.01 --optimizer RAdam --patience 3

# Some additional attempts
python run.py --emb_dim 64 --layers [64,64] --lr 0.003 --lr_scheduler CyclicLR
python run.py --emb_dim 64 --layers [64,64] --lr 0.003 --optimizer RAdam --patience 3

# message dropout
python run.py --emb_dim 64 --layers [64,64] --mess_dropout 0.1 --lr 0.002
python run.py --emb_dim 64 --layers [64,64] --mess_dropout 0.3 --lr 0.003
python run.py --emb_dim 64 --layers [64,64] --mess_dropout 0.5 --lr 0.005

# BPR
python run.py --emb_dim 16 --layers [16,16] --model BPR
python run.py --emb_dim 32 --layers [32,32] --model BPR
python run.py --emb_dim 64 --layers [64,64] --model BPR

# last experiment
python run.py --n_epochs 100 --emb_dim 64 --layers [64,64] --lr_scheduler ReduceLROnPlateau --patience 4 --pretrain 1