class BaseEstimator:
    def __init__(self, model, dt):
        self.model = model
        self.dt = dt
    
    def estimate(self, y, u, validity):
        raise NotImplementedError("estimate method not implemented")