

# python base_main.py \
# --config-file=configs/examples/betavae_shapes3d.yaml \
# --log.project=demo_test \
# --log.group=betavae_shapes3d_lr-0.01 \
# --log.wandb_mode=dryrun \
# --train.seed=0 \
# --train.lr=1e-2 \
# --train.epochs=200 \
# --train.batch_size=5120 \
# --run.do="['train','eval','visualize']" \
# --eval.metrics="['dci_d','reconstruction_error']" \
# --eval.mode=last \
# --viz.show_loss=False \
# --data.num_workers=15 


# python base_main.py \
# --config-file=configs/examples/vae_shapes3d.yaml \
# --log.project=demo_test \
# --log.group=vae_shapes3d_lr-0.01 \
# --log.wandb_mode=dryrun \
# --train.seed=0 \
# --train.lr=1e-2 \
# --train.epochs=200 \
# --train.batch_size=5120 \
# --run.do="['train','eval','visualize']" \
# --eval.metrics="['dci_d','reconstruction_error']" \
# --eval.mode=last \
# --viz.show_loss=False \
# --data.num_workers=15 


python base_main.py \
    --config-file=configs/examples/betavae_shapes3d.yaml \
    --log.project=demo_test \
    --log.group=betavae_shapes3d_lr-0.001-fs \
    --log.wandb_mode=dryrun \
    --train.seed=0 \
    --train.lr=1e-3 \
    --train.epochs=200 \
    --train.batch_size=64 \
    --run.do="['train','eval','visualize']" \
    --eval.metrics="['dci_d','reconstruction_error']" \
    --eval.mode=last \
    --viz.show_loss=False \
    --data.num_workers=15 \
    --train.loss=factorizedsupportvae \
    --factorizedsupportvae.beta=0 \
    --factorizedsupportvae.gamma=300