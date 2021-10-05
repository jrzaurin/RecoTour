# MOVIELENS
python main_pytorch.py --dataset movielens --p_dims "[100,300]" --anneal_cap 1.
python main_pytorch.py --dataset movielens --p_dims "[200,600]" --anneal_cap 1.
python main_pytorch.py --dataset movielens --p_dims "[300,900]" --anneal_cap 1.

python main_mxnet.py --dataset movielens --p_dims "[100,300]" --anneal_cap 1.
python main_mxnet.py --dataset movielens --p_dims "[200,600]" --anneal_cap 1.
python main_mxnet.py --dataset movielens --p_dims "[300,900]" --anneal_cap 1.

# AMAZON
python main_pytorch.py --p_dims "[100,300]" --anneal_cap 1.
python main_pytorch.py --p_dims "[200,600]" --anneal_cap 1.
python main_pytorch.py --p_dims "[300,900]" --anneal_cap 1.

python main_mxnet.py --p_dims "[100,300]" --anneal_cap 1.
python main_mxnet.py --p_dims "[200,600]" --anneal_cap 1.
python main_mxnet.py --p_dims "[300,900]" --anneal_cap 1.

# MOVIELENS
python main_pytorch.py --dataset movielens --p_dims "[100,300]" --anneal_cap 0.6 --save_results
python main_pytorch.py --dataset movielens --p_dims "[200,600]" --constant_anneal --anneal_cap 0. --save_results
python main_pytorch.py --dataset movielens --p_dims "[200,600]" --anneal_cap 0.2 --save_results
python main_pytorch.py --dataset movielens --p_dims "[200,600]" --anneal_cap 0.3 --save_results
python main_pytorch.py --dataset movielens --p_dims "[300,900]" --anneal_cap 0.4 --save_results

python main_mxnet.py --dataset movielens --p_dims "[100,300]" --anneal_cap 0.15 --save_results
python main_mxnet.py --dataset movielens --p_dims "[200,600]" --constant_anneal --anneal_cap 0. --save_results
python main_mxnet.py --dataset movielens --p_dims "[200,600]" --anneal_cap 0.08 --save_results
python main_mxnet.py --dataset movielens --p_dims "[300,900]" --anneal_cap 0.06 --save_results

# AMAZON
python main_pytorch.py --p_dims "[100,300]" --anneal_cap 0.2 --save_results
python main_pytorch.py --p_dims "[200,600]" --constant_anneal --anneal_cap 0. --save_results
python main_pytorch.py --p_dims "[200,600]" --anneal_cap 0.2 --save_results
python main_pytorch.py --p_dims "[200,600]" --anneal_cap 0.8 --save_results
python main_pytorch.py --p_dims "[300,900]" --anneal_cap 0.7 --save_results

python main_mxnet.py --p_dims "[100,300]" --anneal_cap 0.12 --save_results
python main_mxnet.py --p_dims "[200,600]" --constant_anneal --anneal_cap 0. --save_results
python main_mxnet.py --p_dims "[200,600]" --anneal_cap 0.06 --save_results
python main_mxnet.py --p_dims "[300,900]" --anneal_cap 0.05 --save_results

# MOVIELENS
python main_pytorch.py --dataset movielens --p_dims "[50,150]" --model "dae" --save_results
python main_pytorch.py --dataset movielens --p_dims "[100,300]" --model "dae" --save_results
python main_pytorch.py --dataset movielens --p_dims "[200,600]" --model "dae" --save_results
python main_pytorch.py --dataset movielens --p_dims "[300,900]" --model "dae" --save_results

python main_mxnet.py --dataset movielens --p_dims "[50,150]" --model "dae" --save_results
python main_mxnet.py --dataset movielens --p_dims "[100,300]" --model "dae" --save_results
python main_mxnet.py --dataset movielens --p_dims "[200,600]" --model "dae" --save_results
python main_mxnet.py --dataset movielens --p_dims "[300,900]" --model "dae" --save_results

# AMAZON
python main_pytorch.py --p_dims "[50,150]" --model "dae" --save_results
python main_pytorch.py --p_dims "[100,300]" --model "dae" --save_results
python main_pytorch.py --p_dims "[200,600]" --model "dae" --save_results
python main_pytorch.py --p_dims "[300,900]" --model "dae" --save_results

python main_mxnet.py --p_dims "[50,150]" --model "dae" --save_results
python main_mxnet.py --p_dims "[100,300]" --model "dae" --save_results
python main_mxnet.py --p_dims "[200,600]" --model "dae" --save_results
python main_mxnet.py --p_dims "[300,900]" --model "dae" --save_results

# MOVIELENS
python main_pytorch.py --dataset movielens --p_dims "[256,512,1024]" --dropout_enc "[0.5,0.2,0.2]" --dropout_dec "[0.2,0.2,0.2]" --save_results
python main_pytorch.py --dataset movielens --p_dims "[256,512,1024]" --dropout_enc "[0.5,0.,0.]" --dropout_dec "[0.,0.,0.]" --save_results
python main_pytorch.py --dataset movielens --lr 0.005 --early_stop_patience 30 --lr_scheduler --lr_patience 20 --save_results
python main_pytorch.py --dataset movielens --weight_decay 0.001 --save_results

python main_mxnet.py --dataset movielens --model "dae" --p_dims "[256,512,1024]" --dropout_enc "[0.5,0.2,0.2]" --dropout_dec "[0.2,0.2,0.2]" --save_results
python main_mxnet.py --dataset movielens --model "dae" --p_dims "[256,512,1024]" --dropout_enc "[0.5,0.,0.]" --dropout_dec "[0.,0.,0.]" --save_results
python main_mxnet.py --dataset movielens --model "dae" --lr 0.005 --early_stop_patience 30 --lr_scheduler --lr_patience 20 --save_results
python main_mxnet.py --dataset movielens --model "dae" --weight_decay 0.001 --save_results

# AMAZON
python main_pytorch.py --p_dims "[256,512,1024]" --dropout_enc "[0.5,0.2,0.2]" --dropout_dec "[0.2,0.2,0.2]" --anneal_cap 0.4 --save_results
python main_pytorch.py --p_dims "[256,512,1024]" --dropout_enc "[0.5,0.,0.]" --dropout_dec "[0.,0.,0.]" --anneal_cap 0.4 --save_results
python main_pytorch.py --p_dims "[300,900]" --lr 0.005 --early_stop_patience 30 --lr_scheduler --lr_patience 20 --anneal_cap 0.4 --save_results
python main_pytorch.py --p_dims "[300,900]" --weight_decay 0.001 --anneal_cap 0.4 --save_results

python main_mxnet.py --model "dae" --p_dims "[256,512,1024]" --dropout_enc "[0.5,0.2,0.2]" --dropout_dec "[0.2,0.2,0.2]" --save_results
python main_mxnet.py --model "dae" --p_dims "[256,512,1024]" --dropout_enc "[0.5,0.,0.]" --dropout_dec "[0.,0.,0.]" --save_results
python main_mxnet.py --model "dae" --p_dims "[100,300]" --lr 0.005 --early_stop_patience 30 --lr_scheduler --lr_patience 20 --save_results
python main_mxnet.py --model "dae" --p_dims "[100,300]" --weight_decay 0.001 --save_results

# MOVIELENS
python main_pytorch.py --dataset movielens --dropout_enc "[0.2, 0.]" --weight_decay 0.001 --model "dae" --save_results
python main_pytorch.py --dataset movielens --dropout_enc "[0.2, 0.]" --weight_decay 0.001 --lr 0.005 --early_stop_patience 30 --lr_scheduler --lr_patience 20 --save_results
python main_mxnet.py --dataset movielens --dropout_enc "[0.2, 0.]" --weight_decay 0.001 --model "dae" --lr 0.005 --early_stop_patience 30 --lr_scheduler --lr_patience 20 --save_results
python main_mxnet.py --dataset movielens --dropout_enc "[0.2, 0.]" --weight_decay 0.001 --lr 0.005 --early_stop_patience 30 --lr_scheduler --lr_patience 20 --save_results

# AMAZON
python main_pytorch.py --dropout_enc "[0.2, 0.]" --weight_decay 0.001 --p_dims "[50,150]" --model "dae" --save_results
python main_pytorch.py --dropout_enc "[0.2, 0.]" --weight_decay 0.001 --p_dims "[300,900]" --anneal_cap 0.5 --save_results
python main_mxnet.py --dropout_enc "[0.2, 0.]" --weight_decay 0.001 --p_dims "[100,300]" --model "dae" --save_results
python main_mxnet.py --dropout_enc "[0.2, 0.]" --weight_decay 0.001 --lr 0.005 --early_stop_patience 30 --lr_scheduler --lr_patience 20 --save_results
