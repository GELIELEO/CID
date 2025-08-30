
seeds=(42)

for seed in "${seeds[@]}"; do
    python base_main.py --config-file=configs/montero_exps/montero_betavae_shapes3d.yaml --log.wandb_mode=off --train.epochs=50 --log.project=baseline-pair1 --train.seed=$seed --constraints.correlations_file=constraints/avail_correlations.yaml:shapes3d_single_1_01 --run.gpu_id=1
    sleep 5

    python base_main.py --config-file=configs/montero_exps/montero_betavae_shapes3d.yaml --log.wandb_mode=off --train.epochs=50 --log.project=baseline-pair2 --train.seed=$seed --constraints.correlations_file=constraints/avail_correlations.yaml:shapes3d_double_1_01 --run.gpu_id=1
    sleep 5

    python base_main.py --config-file=configs/montero_exps/montero_betavae_shapes3d.yaml --log.wandb_mode=off --train.epochs=50 --log.project=baseline-pair3 --train.seed=$seed --constraints.correlations_file=constraints/avail_correlations.yaml:shapes3d_triple_1_01 --run.gpu_id=1
    sleep 5
    
    python base_main.py --config-file=configs/montero_exps/montero_betavae_shapes3d.yaml --log.wandb_mode=off --train.epochs=50 --log.project=baseline-conf --train.seed=$seed --constraints.correlations_file=constraints/avail_correlations.yaml:shapes3d_conf_1_02 --run.gpu_id=1
    sleep 5
done

for seed in "${seeds[@]}"; do
    python base_main.py --config-file=configs/montero_exps/montero_betavae_shapes3d.yaml --log.wandb_mode=off --train.epochs=50 --log.project=baseline-pair1 --log.group=betavae_shapes3d-hfs --train.epochs=50 --train.seed=$seed --train.loss=factorizedsupportvae --factorizedsupportvae.beta=0 --factorizedsupportvae.gamma=300 --constraints.correlations_file=constraints/avail_correlations.yaml:shapes3d_single_1_01 --run.gpu_id=1
    sleep 5

    python base_main.py --config-file=configs/montero_exps/montero_betavae_shapes3d.yaml --log.wandb_mode=off --train.epochs=50 --log.project=baseline-pair2 --log.group=betavae_shapes3d-hfs --train.epochs=50 --train.seed=$seed --train.loss=factorizedsupportvae --factorizedsupportvae.beta=0 --factorizedsupportvae.gamma=300 --constraints.correlations_file=constraints/avail_correlations.yaml:shapes3d_double_1_01 --run.gpu_id=1
    sleep 5

    python base_main.py --config-file=configs/montero_exps/montero_betavae_shapes3d.yaml --log.wandb_mode=off --train.epochs=50 --log.project=baseline-pair3 --log.group=betavae_shapes3d-hfs --train.epochs=50 --train.seed=$seed --train.loss=factorizedsupportvae --factorizedsupportvae.beta=0 --factorizedsupportvae.gamma=300 --constraints.correlations_file=constraints/avail_correlations.yaml:shapes3d_triple_1_01 --run.gpu_id=1
    sleep 5
    
    python base_main.py --config-file=configs/montero_exps/montero_betavae_shapes3d.yaml --log.wandb_mode=off --train.epochs=50 --log.project=baseline-conf --log.group=betavae_shapes3d-hfs --train.epochs=50 --train.seed=$seed --train.loss=factorizedsupportvae --factorizedsupportvae.beta=0 --factorizedsupportvae.gamma=300 --constraints.correlations_file=constraints/avail_correlations.yaml:shapes3d_conf_1_02 --run.gpu_id=1
    sleep 5
done