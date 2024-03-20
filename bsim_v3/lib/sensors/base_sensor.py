class BaseSensor:
    _num_outputs = -1

    @property
    def num_outputs(self):
        if self._num_outputs == -1:
            raise NotImplementedError("num_outputs not implemented")

        return self._num_outputs

    def get_output(self, state):
        raise NotImplementedError("get_output method not implemented")

    def subtract_outputs(self, o1, o2):
        raise NotImplementedError("subtract_outputs method not implemented")
    
    def output_mean(self, sigmas, Wm):
        raise NotImplementedError("output_mean method not implemented")