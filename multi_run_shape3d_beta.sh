
seeds=(0 42 1993 2020)

for seed in "${seeds[@]}"; do
    # python base_main.py --config-file=configs/montero_exps/intervene_montero_betavae_shape3d.yaml --log.wandb_mode=off --train.epochs=5 --log.project=me0.996md0.996wd-beta1.0-bs128 --train.seed=$seed --rect.rect_coeff=1.0 --betavae.beta=1.0 --shuffled_betavae.shuffled_coeff=0.0 --run.do="['train']" --train.batch_size=128 --train.lr=1e-3 --train.checkpoint_every=5 --scheduler.name=none --run.gpu_id=0 
    # sleep 5

    python base_main.py --config-file=configs/montero_exps/intervene_montero_betavae_shape3d.yaml --log.wandb_mode=off --train.epochs=50 --log.project=me0.996md0.996js-beta1.0-bs128 --train.seed=$seed --rect.rect_coeff=1.0 --betavae.beta=1.0 --shuffled_betavae.shuffled_coeff=5.0 --train.batch_size=128 --train.lr=2e-3 --run.gpu_id=0
    sleep 5
done