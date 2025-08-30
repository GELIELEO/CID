
momentum=(0 0.1 0.3 0.5 0.7)

for m in "${momentum[@]}"; do

    # python base_main.py --config-file=configs/montero_exps/montero_betavae_shapes3d.yaml --log.wandb_mode=off --log.project=baseline --train.epochs=50 --train.seed=$seed
    # sleep 5

    python base_main.py --config-file=configs/montero_exps/intervene_montero_betavae_shape3d.yaml --log.wandb_mode=off --train.epochs=50 --log.project=ablation --log.group=intervene_betavae_shape3d_momentum$m --shuffled_betavae.momentum=$m --rect.rect_coeff=1.0 --shuffled_betavae.shuffled_coeff=2.0 --train.batch_size=128 --train.lr=1e-3  --run.gpu_id=1

    # python base_main.py --config-file=configs/montero_exps/montero_betavae_dsprites.yaml --log.wandb_mode=off --train.epochs=50 --train.seed=$seed

done