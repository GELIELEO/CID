
seeds=(0 42 1993 2020)

for seed in "${seeds[@]}"; do
    # python base_main.py --config-file=configs/montero_exps/intervene_montero_betavae_shape3d.yaml --log.wandb_mode=off --train.epochs=5 --log.project=me0.996md0.996wd-beta0.0-bs128 --train.seed=$seed --rect.rect_coeff=1.0 --shuffled_betavae.shuffled_coeff=0.0 --run.do="['train']" --train.batch_size=128 --train.lr=1e-3 --train.checkpoint_every=5 --scheduler.name=none
    # sleep 5

    python base_main.py --config-file=configs/montero_exps/intervene_montero_betavae_mpi3d.yaml --log.wandb_mode=off --train.epochs=50 --log.project=me0.996md0.996js-beta0.0-bs128-lr1e3 --train.seed=$seed --rect.rect_coeff=1.0 --shuffled_betavae.shuffled_coeff=0.5 --train.batch_size=128 --train.lr=1e-3
    sleep 5
done