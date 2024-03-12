class BasePlant:
    def __init__(self, x0, dt):
        self.x = x0
        self.dt = dt

    def next(self):
        raise NotImplementedError("next method not implemented")
    
    def get_state(self):
        return self.x
