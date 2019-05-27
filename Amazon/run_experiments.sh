python gmf.py --batch_size 512 --lr 0.01 --n_emb 8 --epochs 30
python gmf.py --batch_size 1024 --lr 0.01 --n_emb 8 --epochs 30

python gmf.py --batch_size 512 --lr 0.01 --n_emb 16 --epochs 30 --validate_every 2

python gmf.py --batch_size 512 --lr 0.01 --lr_scheduler --learner "SGD" --n_emb 8 --epochs 30
python gmf.py --batch_size 512 --lr 0.01 --lr_scheduler --learner "RMSprop" --n_emb 8 --epochs 30
python gmf.py --batch_size 512 --lr 0.01 --lr_scheduler --n_emb 8 --epochs 30

python gmf.py --batch_size 512 --lr 0.001 --n_emb 64 --epochs 30

python mlp.py --batch_size 512 --lr 0.01 --layers "[32, 16, 8]" --epochs 30
python mlp.py --batch_size 512 --lr 0.01 --layers "[64, 32, 16]" --epochs 30
python mlp.py --batch_size 512 --lr 0.01 --layers "[128, 64, 32]" --epochs 30
python mlp.py --batch_size 512 --lr 0.01 --layers "[256, 128, 64]" --epochs 30

python gmf.py --batch_size 512 --lr 0.01 --n_emb 64 --epochs 30
python gmf.py --batch_size 512 --lr 0.03 --n_emb 64 --lr_scheduler --epochs 30
python gmf.py --batch_size 512 --lr 0.03 --n_emb 64 --epochs 30

python mlp.py --batch_size 512 --lr 0.03 --layers "[32, 16, 8]" --epochs 30
python mlp.py --batch_size 512 --lr 0.03 --layers "[64, 32, 16]" --epochs 30
python mlp.py --batch_size 512 --lr 0.03 --layers "[128, 64, 32]" --epochs 30
python mlp.py --batch_size 512 --lr 0.03 --layers "[256, 128, 64]" --epochs 30

python gmf.py --batch_size 512 --lr 0.03 --n_emb 8 --epochs 30
python gmf.py --batch_size 512 --lr 0.03 --n_emb 16 --epochs 30
python gmf.py --batch_size 512 --lr 0.03 --n_emb 32 --epochs 30

python gmf.py --batch_size 512 --lr 0.03 --n_emb 32 --epochs 30

python neumf.py --batch_size 512 --lr 0.03 --lr_scheduler --n_emb 32 --layers "[128, 64, 32]" --dropouts "[0.,0.]" \
--mf_pretrain "GMF_bs_512_lr_003_n_emb_32_lrnr_adam_lrs_wolrs.pt" \
--mlp_pretrain "MLP_bs_512_reg_00_lr_003_n_emb_64_ll_32_dp_wodp_lrnr_adam_lrs_wolrs.pt" \
--epochs 20

python neumf.py --batch_size 512 --lr 0.003 --n_emb 32 --layers "[128, 64, 32]" --dropouts "[0.,0.]" \
--mf_pretrain "GMF_bs_512_lr_003_n_emb_32_lrnr_adam_lrs_wolrs.pt" \
--mlp_pretrain "MLP_bs_512_reg_00_lr_003_n_emb_64_ll_32_dp_wodp_lrnr_adam_lrs_wolrs.pt" \
--freeze 1 --epochs 6

python neumf.py --batch_size 512 --lr 0.01 --lr_scheduler --n_emb 32 --layers "[128, 64, 32]" --dropouts "[0.,0.]" \
--mf_pretrain "GMF_bs_512_lr_003_n_emb_32_lrnr_adam_lrs_wolrs.pt" \
--mlp_pretrain "MLP_bs_512_reg_00_lr_003_n_emb_64_ll_32_dp_wodp_lrnr_adam_lrs_wolrs.pt" \
--freeze 1 --epochs 6