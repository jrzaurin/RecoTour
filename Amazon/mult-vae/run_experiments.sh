python main_pytorch.py --dataset movielens --save_results
python main_pytorch.py --dataset movielens --constant_anneal --anneal_cap 0.0 --save_results
python main_pytorch.py --dataset movielens --p_dims "[128, 512]" --dropout_enc "[0.5, 0]" --dropout_dec "[0., 0.]" --lr_scheduler --save_results
python main_pytorch.py --dataset movielens --p_dims "[256, 512]" --dropout_enc "[0.5, 0]" --dropout_dec "[0., 0.]" --lr_scheduler --save_results
python main_pytorch.py --dataset movielens --p_dims "[256, 512]" --dropout_enc "[0.5, 0.2]" --dropout_dec "[0.2, 0.2]" --lr_scheduler --save_results
python main_pytorch.py --dataset movielens --p_dims "[256, 1024]" --dropout_enc "[0.5, 0.2]" --dropout_dec "[0.2, 0.2]" --lr_scheduler --save_results
python main_pytorch.py --dataset movielens --p_dims "[256, 512, 1024]" --dropout_enc "[0.5, 0.2, 0.2]" --dropout_dec "[0.2, 0.2, 0.2]" --lr_scheduler --save_results

python main_mxnet.py --dataset movielens --save_results
python main_mxnet.py --dataset movielens  --constant_anneal --anneal_cap 0.0 --save_results
python main_mxnet.py --dataset movielens --p_dims "[128, 512]" --dropout_enc "[0.5, 0]" --dropout_dec "[0., 0.]" --lr_scheduler --save_results
python main_mxnet.py --dataset movielens --p_dims "[256, 512]" --dropout_enc "[0.5, 0]" --dropout_dec "[0., 0.]" --lr_scheduler --save_results
python main_mxnet.py --dataset movielens --p_dims "[256, 512]" --dropout_enc "[0.5, 0.2]" --dropout_dec "[0.2, 0.2]" --lr_scheduler --save_results
python main_mxnet.py --dataset movielens --p_dims "[256, 1024]" --dropout_enc "[0.5, 0.2]" --dropout_dec "[0.2, 0.2]" --lr_scheduler --save_results
python main_mxnet.py --dataset movielens --p_dims "[256, 512, 1024]" --dropout_enc "[0.5, 0.2, 0.2]" --dropout_dec "[0.2, 0.2, 0.2]" --lr_scheduler --save_results

python main_pytorch.py --dataset movielens --weight_decay 0.001 --save_results
python main_pytorch.py --dataset movielens --p_dims "[512, 1024]" --save_results

python main_mxnet.py --dataset movielens --weight_decay 0.001 --save_results
python main_mxnet.py --dataset movielens --p_dims "[512, 1024]" --save_results
python main_mxnet.py --dataset movielens --anneal_cap 0.05 --save_results
python main_mxnet.py --dataset movielens --anneal_cap 0.1 --lr 0.0001 --save_results

python main_pytorch.py --dataset movielens --model "dae" --save_results
python main_pytorch.py --dataset movielens --model "dae" --p_dims "[128, 256]" --save_results
python main_pytorch.py --dataset movielens --model "dae" --p_dims "[256, 512]" --save_results
python main_pytorch.py --dataset movielens --model "dae" --p_dims "[512, 1024]" --save_results

python main_mxnet.py --dataset movielens --model "dae" --save_results
python main_mxnet.py --dataset movielens --model "dae" --p_dims "[128, 256]" --save_results
python main_mxnet.py --dataset movielens --model "dae" --p_dims "[256, 512]" --save_results
python main_mxnet.py --dataset movielens --model "dae" --p_dims "[512, 1024]" --save_results

python main_pytorch.py --dataset movielens --model "vae" --p_dims "[128, 256]" --save_results
python main_pytorch.py --dataset movielens --model "vae" --p_dims "[256, 512]" --save_results
python main_pytorch.py --dataset movielens --model "vae" --p_dims "[512, 1024]" --early_stop_patience 30 --lr_scheduler --lr_patience 10 --save_results

python main_mxnet.py --dataset movielens --model "vae" --anneal_cap 0.02 --save_results
python main_mxnet.py --dataset movielens --model "vae" --anneal_cap 0.02 --p_dims "[128, 256]" --save_results
python main_mxnet.py --dataset movielens --model "vae" --anneal_cap 0.02 --p_dims "[256, 512]" --save_results
python main_mxnet.py --dataset movielens --model "vae" --anneal_cap 0.02 --p_dims "[512, 1024]" --save_results --early_stop_patience 30 --lr_scheduler --lr_patience 10 --save_results

python main_pytorch.py --anneal_cap 0. --constant_anneal --save_results
python main_pytorch.py --anneal_cap 0.1 --save_results
python main_pytorch.py --anneal_cap 0.2 --save_results

python main_mxnet.py --anneal_cap 0. --constant_anneal --save_results
python main_mxnet.py --anneal_cap 0.1 --save_results
python main_mxnet.py --anneal_cap 0.2 --save_results

python main_mxnet.py --anneal_cap 0. --constant_anneal --p_dims "[64,128]" --save_results
python main_mxnet.py --anneal_cap 0. --constant_anneal --p_dims "[64,128]" --lr 0.01 --save_results
python main_mxnet.py --anneal_cap 0. --constant_anneal --p_dims "[64,256]" --save_results
python main_mxnet.py --anneal_cap 0. --constant_anneal --p_dims "[128,512]" --save_results
python main_mxnet.py --anneal_cap 0. --constant_anneal --p_dims "[512,1024]" --early_stop_patience 10 --save_results

python main_pytorch.py --model "dae" --save_results
python main_mxnet.py --model "dae" --p_dims "[64,256]" --save_results

# ###############################################################################
# MOVIELENS
python main_pytorch.py --dataset movielens --p_dims "[100,300]" --save_results
python main_pytorch.py --dataset movielens --p_dims "[200,600]" --save_results
python main_pytorch.py --dataset movielens --p_dims "[300,900]" --save_results

python main_mxnet.py --dataset movielens --p_dims "[100,300]" --save_results
python main_mxnet.py --dataset movielens --p_dims "[200,600]" --save_results
python main_mxnet.py --dataset movielens --p_dims "[300,900]" --save_results

python main_pytorch.py --dataset movielens --model "dae" --p_dims "[100,300]" --save_results
python main_pytorch.py --dataset movielens --model "dae" --p_dims "[200,600]" --save_results
python main_pytorch.py --dataset movielens --model "dae" --p_dims "[300,900]" --save_results

python main_mxnet.py --dataset movielens --model "dae" --p_dims "[100,300]" --save_results
python main_mxnet.py --dataset movielens --model "dae" --p_dims "[200,600]" --save_results
python main_mxnet.py --dataset movielens --model "dae" --p_dims "[300,900]" --save_results

python main_pytorch.py --dataset movielens --p_dims "[200,600]" --anneal_cap 0. --constant_anneal --save_results
python main_pytorch.py --dataset movielens --p_dims "[200,600]" --anneal_cap 0.2 --save_results
python main_pytorch.py --dataset movielens --p_dims "[200,600]" --anneal_cap 0.4 --save_results
python main_pytorch.py --dataset movielens --p_dims "[200,600]" --anneal_cap 0.6 --save_results
python main_pytorch.py --dataset movielens --p_dims "[200,600]" --anneal_cap 0.8 --save_results
python main_pytorch.py --dataset movielens --p_dims "[200,600]" --anneal_cap 1.0 --save_results

python main_mxnet.py --dataset movielens --p_dims "[200,600]" --anneal_cap 0. --constant_anneal --save_results
python main_mxnet.py --dataset movielens --p_dims "[200,600]" --anneal_cap 0.2 --save_results
python main_mxnet.py --dataset movielens --p_dims "[200,600]" --anneal_cap 0.4 --save_results
python main_mxnet.py --dataset movielens --p_dims "[200,600]" --anneal_cap 0.6 --save_results
python main_mxnet.py --dataset movielens --p_dims "[200,600]" --anneal_cap 0.8 --save_results
python main_mxnet.py --dataset movielens --p_dims "[200,600]" --anneal_cap 1.0 --save_results

# AMAZON
python main_pytorch.py --p_dims "[50,150]" --save_results
python main_pytorch.py --p_dims "[100,300]" --save_results
python main_pytorch.py --p_dims "[200,600]" --save_results

python main_mxnet.py --p_dims "[50,150]" --save_results
python main_mxnet.py --p_dims "[100,300]" --save_results
python main_mxnet.py --p_dims "[200,600]" --save_results

python main_pytorch.py --model "dae" --p_dims "[50,150]" --save_results
python main_pytorch.py --model "dae" --p_dims "[100,300]" --save_results
python main_pytorch.py --model "dae" --p_dims "[200,600]" --save_results

python main_mxnet.py --model "dae" --p_dims "[50,150]" --save_results
python main_mxnet.py --model "dae" --p_dims "[100,300]" --save_results
python main_mxnet.py --model "dae" --p_dims "[200,600]" --save_results

python main_pytorch.py --p_dims "[100,300]" --anneal_cap 0. --constant_anneal --save_results
python main_pytorch.py --p_dims "[100,300]" --anneal_cap 0.2 --save_results
python main_pytorch.py --p_dims "[100,300]" --anneal_cap 0.4 --save_results
python main_pytorch.py --p_dims "[100,300]" --anneal_cap 0.6 --save_results
python main_pytorch.py --p_dims "[100,300]" --anneal_cap 0.8 --save_results
python main_pytorch.py --p_dims "[100,300]" --anneal_cap 1.0 --save_results

python main_mxnet.py --p_dims "[100,300]" --anneal_cap 0. --constant_anneal --save_results
python main_mxnet.py --p_dims "[100,300]" --anneal_cap 0.2 --save_results
python main_mxnet.py --p_dims "[100,300]" --anneal_cap 0.4 --save_results
python main_mxnet.py --p_dims "[100,300]" --anneal_cap 0.6 --save_results
python main_mxnet.py --p_dims "[100,300]" --anneal_cap 0.8 --save_results
python main_mxnet.py --p_dims "[100,300]" --anneal_cap 1.0 --save_results

MOVIELENS
python main_pytorch.py --dataset movielens --p_dims "[100,300]" --n_epochs 200 --anneal_cap 1.
python main_pytorch.py --dataset movielens --p_dims "[200,600]" --n_epochs 200 --anneal_cap 1.
python main_pytorch.py --dataset movielens --p_dims "[300,900]" --n_epochs 200 --anneal_cap 1.

python main_mxnet.py --dataset movielens --p_dims "[100,300]" --n_epochs 200 --anneal_cap 1.
python main_mxnet.py --dataset movielens --p_dims "[200,600]" --n_epochs 200 --anneal_cap 1.
python main_mxnet.py --dataset movielens --p_dims "[300,900]" --n_epochs 200 --anneal_cap 1.

python main_pytorch.py --dataset movielens --p_dims "[100,300]" --anneal_cap 0.62 --save_results
python main_pytorch.py --dataset movielens --p_dims "[200,600]" --anneal_cap 0.45 --save_results
python main_pytorch.py --dataset movielens --p_dims "[300,900]" --anneal_cap 0.40 --save_results

python main_mxnet.py --dataset movielens --p_dims "[100,300]" --anneal_cap 0.13 --save_results
python main_mxnet.py --dataset movielens --p_dims "[200,600]" --anneal_cap 0.07 --save_results
python main_mxnet.py --dataset movielens --p_dims "[300,900]" --anneal_cap 0.06 --save_results

AMAZON
python main_pytorch.py --p_dims "[100,300]" --n_epochs 200 --anneal_cap 1.
python main_pytorch.py --p_dims "[200,600]" --n_epochs 200 --anneal_cap 1.
python main_pytorch.py --p_dims "[300,900]" --n_epochs 200 --anneal_cap 1.

python main_mxnet.py --p_dims "[100,300]" --n_epochs 200 --anneal_cap 1.
python main_mxnet.py --p_dims "[200,600]" --n_epochs 200 --anneal_cap 1.
python main_mxnet.py --p_dims "[300,900]" --n_epochs 200 --anneal_cap 1.

python main_pytorch.py --p_dims "[100,300]" --anneal_cap 1.0 --save_results
python main_pytorch.py --p_dims "[200,600]" --anneal_cap 0.3 --save_results
python main_pytorch.py --p_dims "[300,900]" --anneal_cap 0.8 --save_results

python main_mxnet.py --p_dims "[100,300]" --anneal_cap 0.10 --save_results
python main_mxnet.py --p_dims "[200,600]" --anneal_cap 0.06 --save_results
python main_mxnet.py --p_dims "[300,900]" --anneal_cap 0.04 --save_results
