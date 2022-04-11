import numpy as np

class GPTransitionModel():
    def __init__(
        self,
        gp_model_spec,
        inference_strategy,
        optimizer,
        reinitialize=True
    ):
        self.gp_model_spec = gp_model_spec
        self.inference_strategy = inference_strategy
        self.optimizer = optimizer
        self.reinitialize = reinitialize
        self.gp_model = None

    def step(self, obs, action):
        return self.inference_strategy.step(
            obs=obs,
            action=action,
            model=self.gp_model,
        )

    def get_gp_data(self, experience, flatten=True):
        #TODO: extent to batches of experience, right now experience is a long tuple of lists
        obs_tm1, a_tm1, _, _, obs_t = experience
        inputs = np.concatenate(arrays=[obs_tm1, a_tm1], axis=1)
        targets = obs_t - obs_tm1
        if flatten is True:
            inputs = tf.reshape(inputs, (-1, inputs.shape[-1]))
            targets = tf.reshape(targets, (-1, targets.shape[-1]))
        return inputs, targets

    #TODO: refactor to allow for InternalDataTrainingLossMixin
    def loss_closure(self, data, compile=True):
        return self.gp_model.training_loss_closure(data, compile)

    def train(self, experience):
        inputs, targets = self.get_gp_data(experience)
        #TODO: understand why reinitialize model
        #TODO: understand the purpose for Internal/ExternalDataTrainingLossMixin
        if self.gp_model is None or reinitialize:
            self.gp_model = initialize_gp_model(
                data=(inputs, targets),
                model_spec=self.gp_model_spec
            )
        variables = self.gp_model.trainable_variables
        loss = self.loss_closure(data=(inputs, targets))
        return self.optimizer.minimize(loss, variables)
