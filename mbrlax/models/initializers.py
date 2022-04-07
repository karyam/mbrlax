from typing import NamedTuple, Type, Callable


class GPModelSpec(NamedTuple):
    type: Type
    num_inducing: int = 32
    likelihood: Likelihood = None
    prior: Callable = None
    mean_function: Callable
    model_uncertainty = False,
    invlink: Callable


def initialize_gp_model(data, model_spec):
    gp_model = model_spec.type.initialize(
        data=data,
        num_inducing=model_spec.num_inducing,
        likelihood=model_spec.likelihood,
        mean_function=model_spec.mean_function,
    )

    if gp_model.q_mu.shape[-2] >= len(data[0]):
        set_trainable(gp_model.inducing_variable, False)

    if not model_spec.model_uncertainty:
        set_trainable(gp_model.q_sqrt, False)
        for kernel in gp_model.kernels:
            set_trainable(kernel.variance, False)
        gp_model = KernelRegressor(model=gp_model)

    if gp_model_spec.invlink is not None:
        gp_model = InverseLinkWrapper(model=gp_model, invlink=model_spec.invlink)

    return gp_model
