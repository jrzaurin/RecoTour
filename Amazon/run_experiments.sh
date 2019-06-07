# batch size (I run this with MSE)
python gmf.py --batch_size 512 --lr 0.01 --n_emb 8 --epochs 30
python gmf.py --batch_size 1024 --lr 0.01 --n_emb 8 --epochs 30

# MSE vs BCE -> BCE wins, from now on BCE (as expected)
python gmf.py --batch_size 1024 --lr 0.01 --n_emb 8 --epochs 30 --validate_every 2

# learning rates
python gmf.py --batch_size 1024 --lr 0.001 --n_emb 8 --epochs 30 --validate_every 2
python gmf.py --batch_size 1024 --lr 0.005 --n_emb 8 --epochs 30 --validate_every 2
python gmf.py --batch_size 1024 --lr 0.01 --n_emb 8 --lr_scheduler --epochs 30 --validate_every 2

# Embeddings
python gmf.py --batch_size 1024 --lr 0.01 --n_emb 16 --epochs 30 --validate_every 2
python gmf.py --batch_size 1024 --lr 0.01 --n_emb 32 --epochs 30 --validate_every 2
python gmf.py --batch_size 1024 --lr 0.01 --n_emb 64 --epochs 30 --validate_every 2

# batch size
python mlp.py --batch_size 512 --lr 0.01 --layers "[32, 16, 8]" --epochs 30 --validate_every 2
python mlp.py --batch_size 1024 --lr 0.01 --layers "[32, 16, 8]" --epochs 30 --validate_every 2

# learning rates
python mlp.py --batch_size 1024 --lr 0.001 --layers "[32, 16, 8]" --epochs 30 --validate_every 2
python mlp.py --batch_size 1024 --lr 0.005 --layers "[32, 16, 8]" --epochs 30 --validate_every 2
python mlp.py --batch_size 1024 --lr 0.01 --layers "[32, 16, 8]" --epochs 30 --lr_scheduler --validate_every 2

# Embeddings
python mlp.py --batch_size 1024 --lr 0.01 --layers "[64, 32, 16]" --epochs 30 --validate_every 2
python mlp.py --batch_size 1024 --lr 0.01 --layers "[128, 64, 32]" --epochs 30 --validate_every 2

# higher lr and lr_scheduler
python mlp.py --batch_size 1024 --lr 0.03 --layers "[64, 32, 16]" --epochs 30 --validate_every 2
python mlp.py --batch_size 1024 --lr 0.03 --layers "[128, 64, 32]" --epochs 30 --validate_every 2
python mlp.py --batch_size 1024 --lr 0.03 --layers "[64, 32, 16]" --epochs 30 --lr_scheduler --validate_every 2
python mlp.py --batch_size 1024 --lr 0.03 --layers "[128, 64, 32]" --epochs 30 --lr_scheduler --validate_every 2

# I repeated this experiment 3 times: SGD with and without momentum, and a
# third time with MSE
python neumf.py --batch_size 1024 --lr 0.01 --n_emb 8 --lr_scheduler --layers "[128, 64, 32]" --dropouts "[0.,0.]" \
--mf_pretrain "GMF_bs_1024_lr_001_n_emb_8_lrnr_adam_lrs_wolrs.pt" \
--mlp_pretrain "MLP_bs_1024_reg_00_lr_003_n_emb_64_ll_32_dp_wodp_lrnr_adam_lrs_wlrs.pt" \
--epochs 20 --learner "SGD" --validate_every 2

python neumf.py --batch_size 1024 --lr 0.01 --n_emb 8 --lr_scheduler --layers "[128, 64, 32]" --dropouts "[0.,0.]" \
--mf_pretrain "GMF_bs_1024_lr_001_n_emb_8_lrnr_adam_lrs_wolrs.pt" \
--mlp_pretrain "MLP_bs_1024_reg_00_lr_003_n_emb_64_ll_32_dp_wodp_lrnr_adam_lrs_wlrs.pt" \
--epochs 20 --learner "SGD" --validate_every 2

python neumf.py --batch_size 1024 --lr 0.01 --n_emb 8 --lr_scheduler --layers "[128, 64, 32]" --dropouts "[0.,0.]" \
--mf_pretrain "GMF_bs_1024_lr_001_n_emb_8_lrnr_adam_lrs_wolrs.pt" \
--mlp_pretrain "MLP_bs_1024_reg_00_lr_003_n_emb_64_ll_32_dp_wodp_lrnr_adam_lrs_wlrs.pt" \
--epochs 20 --learner "SGD" --validate_every 2

python neumf.py --batch_size 1024 --lr 0.01 --n_emb 8 --lr_scheduler --layers "[128, 64, 32]" --dropouts "[0.,0.]" \
--mf_pretrain "GMF_bs_1024_lr_001_n_emb_8_lrnr_adam_lrs_wolrs.pt" \
--mlp_pretrain "MLP_bs_1024_reg_00_lr_003_n_emb_64_ll_32_dp_wodp_lrnr_adam_lrs_wlrs.pt" \
--epochs 20 --validate_every 2

# I repeated this experiment 3 times: with and without momentum and a 3rd time
# with MSE but did not save it
python neumf.py --batch_size 1024 --lr 0.001 --n_emb 8 --layers "[128, 64, 32]" --dropouts "[0.,0.]" \
--mf_pretrain "GMF_bs_1024_lr_001_n_emb_8_lrnr_adam_lrs_wolrs.pt" \
--mlp_pretrain "MLP_bs_1024_reg_00_lr_003_n_emb_64_ll_32_dp_wodp_lrnr_adam_lrs_wlrs.pt" \
--freeze 1 --epochs 4 --learner "SGD"

python neumf.py --batch_size 1024 --lr 0.001 --n_emb 8 --layers "[128, 64, 32]" --dropouts "[0.,0.]" \
--mf_pretrain "GMF_bs_1024_lr_001_n_emb_8_lrnr_adam_lrs_wolrs.pt" \
--mlp_pretrain "MLP_bs_1024_reg_00_lr_003_n_emb_64_ll_32_dp_wodp_lrnr_adam_lrs_wlrs.pt" \
--freeze 1 --epochs 4 --learner "SGD"

python neumf.py --batch_size 1024 --lr 0.001 --n_emb 8 --layers "[128, 64, 32]" --dropouts "[0.,0.]" \
--mf_pretrain "GMF_bs_1024_lr_001_n_emb_8_lrnr_adam_lrs_wolrs.pt" \
--mlp_pretrain "MLP_bs_1024_reg_00_lr_003_n_emb_64_ll_32_dp_wodp_lrnr_adam_lrs_wlrs.pt" \
--freeze 1 --epochs 4 --learner "SGD"

python neumf.py --batch_size 1024 --lr 0.001 --n_emb 8 --layers "[128, 64, 32]" --dropouts "[0.,0.]" \
--mf_pretrain "GMF_bs_1024_lr_001_n_emb_8_lrnr_adam_lrs_wolrs.pt" \
--mlp_pretrain "MLP_bs_1024_reg_00_lr_003_n_emb_64_ll_32_dp_wodp_lrnr_adam_lrs_wlrs.pt" \
--freeze 1 --epochs 4


