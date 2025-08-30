
seeds=(0 42 1993 2020)

for seed in "${seeds[@]}"; do

    # python base_main.py --config-file=configs/montero_exps/montero_betavae_shapes3d.yaml --log.wandb_mode=off --log.project=baseline --train.epochs=50 --train.seed=$seed
    # sleep 5

    python base_main.py --config-file=configs/montero_exps/montero_betavae_shapes3d.yaml --log.wandb_mode=off --log.project=baseline --log.group=betavae_shapes3d-hfs --train.epochs=50 --train.seed=$seed --train.loss=factorizedsupportvae --factorizedsupportvae.beta=0 --factorizedsupportvae.gamma=300

    # python base_main.py --config-file=configs/montero_exps/montero_betavae_dsprites.yaml --log.wandb_mode=off --train.epochs=50 --train.seed=$seed

done