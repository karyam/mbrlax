from typing import Any, Callable
from gpjax.models import GPModel

class GPModelWrapper():
  """
  Base class for wrapping GPJax's GPModels with additional features.
  Defaults to accessing the base model's attributes, save for custom
  methods and attributes specified during initialization.
  """
  def __init__(self, model: GPModel, name: str = None, **attrs):
    self.__dict__["_wrapper_attrs"] = ("_model", "_name", "_name_scope")\
                                      + tuple(attrs.keys())
    super().__init__(name=name)
    self._model = model
    for name, obj in attrs.items():
      setattr(self, name, obj)

  def __getattr__(self, name: str) -> Any:
    return getattr(self.model, name)

  def __setattr__(self, name: str, value: Any):
    if name in self._wrapper_attrs:
      self.__dict__[name] = value
    else:
      return setattr(self.model, name, value)

  def __delattr__(self, name: str):
    return delattr(self.model, name)

  @property
  def model(self):
    return self._model


class KernelRegressor(GPModelWrapper):
  def __call__(self, x: object, **kwargs):
    return self.predict_f(x, **kwargs)[0]


class InverseLinkWrapper(GPModelWrapper):
  def __init__(self, model: Callable, invlink: Callable):
    super().__init__(model=model, invlink=invlink)

  def __call__(self, *args, **kwargs):
    return self.invlink(self.model(*args, **kwargs))