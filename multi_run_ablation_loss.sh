

python base_main.py --config-file=configs/montero_exps/intervene_montero_betavae_shape3d.yaml --log.wandb_mode=off --train.epochs=50 --log.project=ablation --log.group=intervene_betavae_shape3d_rectonly --rect.rect_coeff=1.0 --shuffled_betavae.shuffled_coeff=0.0 --train.batch_size=128 --train.lr=1e-3 --run.gpu_id=1


python base_main.py --config-file=configs/montero_exps/intervene_montero_betavae_shape3d.yaml --log.wandb_mode=off --train.epochs=50 --log.project=ablation --log.group=intervene_betavae_shape3d_componly --rect.rect_coeff=0.0 --shuffled_betavae.shuffled_coeff=1.0 --train.batch_size=128 --train.lr=1e-3 --run.gpu_id=1