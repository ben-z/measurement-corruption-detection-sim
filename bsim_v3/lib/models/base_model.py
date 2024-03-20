class BaseModel:
    num_states = -1
    num_inputs = -1

    def next(self, state, dt, u):
        raise NotImplementedError("next method not implemented")

    def subtract_states(self, x, y):
        raise NotImplementedError("subtract_states method not implemented")

    def state_mean(self, sigmas, Wm):
        raise NotImplementedError("state_mean method not implemented")
    