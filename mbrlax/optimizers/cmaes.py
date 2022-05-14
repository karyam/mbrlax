from evosax import Strategy, CMA_ES
from evosax.utils import ESLog

def pack_params(initial_params):
    pass

class CMAOptimiser:
    def __init__(
        self, 
        key,
        fitness_function, 
        num_generations, 
        pop_size, 
        num_params, 
        es_params=None, 
        callback=None
    ):
        self.fitness_function = fitness_function
        self.num_generations = num_generations
        self.strategy = CMA_ES(popsize=pop_size, num_dims=num_params)
        self.es_params = self.strategy.default_params if es_params is None else es_params
        self.callback = callback
        self.state = None
    
    def minimize(self, params, data):
        packed_params = pack_params(params)
        
        es_logging = ESLog(packed_params.size, self.num_generations, top_k=5, maximize=False)
        logger = self.es_logging.initialize()
        
        if self.state is None:
            self.state = self.strategy.initialize(rng, self.es_params)
            self.state["mean"] = packed_params
        
        for gen in range(num_generations):
            rng, rng_ask = jax.random.split(rng, 2)
            params, state = self.strategy.ask(rng_ask, state, es_params)
            unpacked_params = unpack_params(params)
            fitness = fitness_function(unpacked_params, )  
            state = self.strategy.tell(params, fitness, state, es_params)
            log = es_logging.update(self.logger, params, fitness)
            self.callback(gen, log["log_top_1"][gen])
        
        unpacked_params = unpack_params(self.state["best_member"])
        return unpacked_params, self.state["best_fitness"]
        