
seeds=(42 1993 2020 0)

for seed in "${seeds[@]}"; do
    python base_main.py --config-file=configs/montero_exps/intervene_montero_betavae_dsprites.yaml --log.wandb_mode=off --train.epochs=50 --log.project=me0.996md0.996js-beta0.0-bs128-pair1 --train.seed=$seed --rect.rect_coeff=1.0 --shuffled_betavae.shuffled_coeff=2.0 --train.batch_size=128 --train.lr=2e-3 --constraints.correlations_file=constraints/avail_correlations.yaml:dsprites_single_1_01 --run.gpu_id=0 --run.do="['train','eval','visualize']"
    sleep 5

    python base_main.py --config-file=configs/montero_exps/intervene_montero_betavae_dsprites.yaml --log.wandb_mode=off --train.epochs=50 --log.project=me0.996md0.996js-beta0.0-bs128-pair2 --train.seed=$seed --rect.rect_coeff=1.0 --shuffled_betavae.shuffled_coeff=2.0 --train.batch_size=128 --train.lr=2e-3 --constraints.correlations_file=constraints/avail_correlations.yaml:dsprites_double_2_04 --run.gpu_id=0 --run.do="['train','eval','visualize']"
    sleep 5
    
    python base_main.py --config-file=configs/montero_exps/intervene_montero_betavae_dsprites.yaml --log.wandb_mode=off --train.epochs=50 --log.project=me0.996md0.996js-beta0.0-bs128-conf --train.seed=$seed --rect.rect_coeff=1.0 --shuffled_betavae.shuffled_coeff=2.0 --train.batch_size=128 --train.lr=2e-3 --constraints.correlations_file=constraints/avail_correlations.yaml:dsprites_conf_1_02 --run.gpu_id=0 --run.do="['train','eval','visualize']"
    sleep 5
done