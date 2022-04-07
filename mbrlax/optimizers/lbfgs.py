


class LBFGS:
    def minimize(self):
        return minimize(fun=nlb_fn(X, y, sigma_y),
               x0=pack_params(jnp.array([1.0, 1.0]), X_m),
               method='L-BFGS-B',
               jac=True)
