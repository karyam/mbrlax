class Adam():
    def minimize(self):
        opt_state = opt_init(model_params)
        loss_history = []
        for epoch in range(num_epochs):
            for batch in train_dataloader:  # iterate over batches
                x, y = batch
                loss, opt_state = train_step(epoch, opt_state, batch)
                params = get_params(opt_state)
                loss_history.append(loss.item())