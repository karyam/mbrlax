class ReplayBuffer(Callable):
    def __init__(self):
        self.buffer = []
    def add_batch(self, items):
        self.buffer.append(items)
        