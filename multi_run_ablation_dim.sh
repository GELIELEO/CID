
dims=(0 1 2 3 4 5 6 7 8 9)

for shuffle_dim in "${dims[@]}"; do

    # python base_main.py --config-file=configs/montero_exps/montero_betavae_shapes3d.yaml --log.wandb_mode=off --log.project=baseline --train.epochs=50 --train.seed=$seed
    # sleep 5

    python base_main.py --config-file=configs/montero_exps/intervene_montero_betavae_shape3d.yaml --log.wandb_mode=off --train.epochs=50 --log.project=ablation --log.group=intervene_betavae_shape3d_sdim$shuffle_dim --shuffled_betavae.shuffle_dims=$shuffle_dim --rect.rect_coeff=1.0 --shuffled_betavae.shuffled_coeff=2.0 --train.batch_size=128 --train.lr=1e-3

    # python base_main.py --config-file=configs/montero_exps/montero_betavae_dsprites.yaml --log.wandb_mode=off --train.epochs=50 --train.seed=$seed

done