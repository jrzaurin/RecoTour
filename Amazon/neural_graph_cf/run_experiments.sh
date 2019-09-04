# Embeddings
# python run.py --emb_dim 16 --layers [16,16]
# python run.py --emb_dim 32 --layers [32,32]
# python run.py --emb_dim 64 --layers [64,64]

# with lr schedulers starting with a higher learning rate
python run.py --emb_dim 64 --layers [64,64] --lr 0.005 --lr_scheduler ReduceLROnPlateau
python run.py --emb_dim 64 --layers [64,64] --lr 0.01 --lr_scheduler CyclicLR

# # trying RAdam
# python run.py --emb_dim 64 --layers [64,64] --lr 0.01 --optimizer RAdam

# message dropout


# BPR