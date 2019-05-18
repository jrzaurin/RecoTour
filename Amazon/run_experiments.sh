time python gmf.py --batch_size 512 --lr 0.01 --n_emb 8 --epochs 10 --save 0
time python gmf.py --batch_size 512 --lr 0.01 --n_emb 16 --epochs 10 --save 0
time python gmf.py --batch_size 512 --lr 0.01 --n_emb 32 --epochs 10 --save 0
time python gmf.py --batch_size 512 --lr 0.01 --n_emb 64 --epochs 10 --save 0

# python gmf.py --batch_size 1024 --lr 0.01 --n_emb 32 --epochs 30
# python gmf.py --batch_size 1024 --lr 0.01 --n_emb 64 --epochs 30

# python gmf.py --batch_size 512 --lr 0.003 --n_emb 32 --epochs 30
# python gmf.py --batch_size 512 --lr 0.003 --n_emb 64 --epochs 30
# python gmf.py --batch_size 1024 --lr 0.003 --n_emb 32 --epochs 30
# python gmf.py --batch_size 1024 --lr 0.003 --n_emb 64 --epochs 30


# python mlp.py --batch_size 256 --lr 0.01 --layers "[32, 16, 8]" --epochs 30
# python mlp.py --batch_size 256 --lr 0.01 --layers "[32, 16, 8]" --dropouts "[0.25, 0.25]" --epochs 40
# python mlp.py --batch_size 256 --lr 0.01 --layers "[64, 32, 16]" --epochs 30
# python mlp.py --batch_size 256 --lr 0.01 --layers "[64, 32, 16]" --dropouts "[0.25, 0.25]" --epochs 40
# python mlp.py --batch_size 256 --lr 0.01 --layers "[128, 64, 32]" --epochs 30
# python mlp.py --batch_size 256 --lr 0.01 --layers "[128, 64, 32]" --dropouts "[0.25, 0.25]" --epochs 40
# python mlp.py --batch_size 256 --lr 0.01 --layers "[256, 128, 64]" --epochs 30
# python mlp.py --batch_size 256 --lr 0.01 --layers "[256, 128, 64]" --dropouts "[0.25, 0.25]" --epochs 40

# python neumf.py --batch_size 256 --lr 0.01 --n_emb 64 --layers "[256, 128, 64]" --dropouts "[0.,0.]" --epochs 30
# python neumf.py --batch_size 256 --lr 0.001 --n_emb 64 --layers "[256, 128, 64]" --dropouts "[0.,0.]" \
# --mf_pretrain "GMF_bs_256_lr_001_n_emb_64.pt" \
# --mlp_pretrain "MLP_bs_256_reg_00_lr_001_n_emb_128_ll_64_dp_wodp.pt" \
# --learner "SGD" --epochs 10
# python neumf.py --batch_size 256 --lr 0.001 --n_emb 64 --layers "[256, 128, 64]" --dropouts "[0.,0.]" \
# --mf_pretrain "GMF_bs_256_lr_001_n_emb_64.pt" \
# --mlp_pretrain "MLP_bs_256_reg_00_lr_001_n_emb_128_ll_64_dp_wodp.pt" \
# --freeze 1 --learner "SGD" --epochs 5

# python neumf.py --batch_size 256 --lr 0.01 --n_emb 64 --layers "[256, 128, 64]" --dropouts "[0.,0.]" \
# --mf_pretrain "GMF_bs_256_lr_001_n_emb_64.pt" \
# --mlp_pretrain "MLP_bs_256_reg_00_lr_001_n_emb_128_ll_64_dp_wodp.pt" \
# --epochs 20
# python neumf.py --batch_size 256 --lr 0.01 --n_emb 64 --layers "[256, 128, 64]" --dropouts "[0.,0.]" \
# --mf_pretrain "GMF_bs_256_lr_001_n_emb_64.pt" \
# --mlp_pretrain "MLP_bs_256_reg_00_lr_001_n_emb_128_ll_64_dp_wodp.pt" \
# --freeze 1 --epochs 20
