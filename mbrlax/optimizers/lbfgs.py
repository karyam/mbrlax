# import scipy
# from typing import Callable
# import numpy as np
# from jax import jit, value_and_grad

# def pack_params(params):
#     shapes["lengthscales"] = params["kernel"][0]["lengthscales"].shape
#     shapes["kernel_variance"] = params["kernel"][0]["variance"].shape
#     shapes["likelihood_variance"] = params["likelihood"]["variance"].shape
#     shapes["inducing_variable"] = params["inducing_variable"].shape
#     shapes["q_mu"] = params["q_mu"].shape
#     shape["q_sqrt"] = params["q_sqrt"].shape
    
#     for kernel in params["kenrel"]:

#     return jnp.concatenate(), num_kernels, shapes

# def unpack_params(params, num_kernels, num_kernel_params, shapes):
#     idx = 0
#     for i in range(num_kernels):
#         kernel_params = params[idx:num_kernel_params]

#         idx += num_kernel_params

# def value_and_grads(loss_function, data):
#     get_value_and_grad = jit(jax.value_and_grad(loss_function))
    
#     def loss_function_wrapper(params):
#         params = unpack_params(params)
#         value, grads = get_value_and_grad(params, data)
#         return np.array(value), np.array(grads)
    
#     return loss_function_wrapper

# class LBFGS:
#     def __init__(self, callback):
#         self.callback = callback
        
#     def minimize(self, loss_function, init_params, data):
#         return scipy.optimize.minimize(
#             fun=value_and_grads(loss_function, data),
#             x0=pack_params(init_params),
#             method='L-BFGS-B',
#             jac=True
#         )
