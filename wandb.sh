docker run -d -p 8080:8080 \
    --name wandb \
    -v wandb:/root/.wandb \
    -e WANDB_API_KEY=d9410fa7704c85ddd2bdf573f1a09dd171d2472c \
    wandb/local