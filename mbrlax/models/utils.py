from typing import Any, Callable
from gpjax.models import GPModel

class KernelRegressor:
  """RBF network"""
  def __init__(self, model: Callable):
    self.model = model

  def __call__(self, x, **kwargs):
    return self.model(x, **kwargs)[0]

  @property
  def trainable_variables(self):
    variables = self.model.get_params()
    kernel_lengthscales = []
    for i in range(len(variables["kernel"])):
      kernel_lengthscales.append(variables["kernel"][i]["lengthscales"])
    trainable_variables = {
      "kernel_lengthscales": kernel_lengthscales,
      "inducing_variables": variables["inducing_variable"],
      "q_mu": variables["q_mu"]
    }
    return trainable_variables


class InverseLinkWrapper(KernelRegressor):
  """RBF network wrapped inside inverse link function"""
  def __init__(self, model: Callable, invlink: Callable):
    self.model = model
    self.invlink = invlink

  def __call__(self, *args, **kwargs):
    return self.invlink(self.model(*args, **kwargs))
  
