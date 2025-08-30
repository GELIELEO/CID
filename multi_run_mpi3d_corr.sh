
seeds=(42 )

for seed in "${seeds[@]}"; do
    # python base_main.py --config-file=configs/montero_exps/intervene_montero_betavae_mpi3d.yaml --log.wandb_mode=off --train.epochs=5 --log.project=me0.996md0.996js-beta0.0-bs128-pair1 --train.seed=$seed --rect.rect_coeff=1.0 --shuffled_betavae.shuffled_coeff=0.0 --run.do="['train']" --train.batch_size=128 --train.lr=1e-3 --train.checkpoint_every=5 --scheduler.name=none --constraints.file=constraints/avail_correlations.yaml:shapes3d_single_1_01
    # sleep 5

    python base_main.py --config-file=configs/montero_exps/intervene_montero_betavae_mpi3d.yaml --log.wandb_mode=off --train.epochs=50 --log.project=me0.996md0.996js-beta0.0-bs128-pair1 --train.seed=$seed --rect.rect_coeff=1.0 --shuffled_betavae.shuffled_coeff=2.0 --train.batch_size=128 --train.lr=1e-3 --constraints.correlations_file=constraints/avail_correlations.yaml:mpi3d_single_1_01 --run.gpu_id=1
    sleep 5

    python base_main.py --config-file=configs/montero_exps/intervene_montero_betavae_mpi3d.yaml --log.wandb_mode=off --train.epochs=50 --log.project=me0.996md0.996js-beta0.0-bs128-pair2 --train.seed=$seed --rect.rect_coeff=1.0 --shuffled_betavae.shuffled_coeff=2.0 --train.batch_size=128 --train.lr=1e-3 --constraints.correlations_file=constraints/avail_correlations.yaml:mpi3d_double_1_04 --run.gpu_id=1
    sleep 5

    python base_main.py --config-file=configs/montero_exps/intervene_montero_betavae_mpi3d.yaml --log.wandb_mode=off --train.epochs=50 --log.project=me0.996md0.996js-beta0.0-bs128-pair3 --train.seed=$seed --rect.rect_coeff=1.0 --shuffled_betavae.shuffled_coeff=2.0 --train.batch_size=128 --train.lr=1e-3 --constraints.correlations_file=constraints/avail_correlations.yaml:mpi3d_triple_1_01 --run.gpu_id=1
    sleep 5
    
    python base_main.py --config-file=configs/montero_exps/intervene_montero_betavae_mpi3d.yaml --log.wandb_mode=off --train.epochs=50 --log.project=me0.996md0.996js-beta0.0-bs128-conf --train.seed=$seed --rect.rect_coeff=1.0 --shuffled_betavae.shuffled_coeff=2.0 --train.batch_size=128 --train.lr=1e-3 --constraints.correlations_file=constraints/avail_correlations.yaml:mpi3d_conf_1_02 --run.gpu_id=1
    sleep 5
done