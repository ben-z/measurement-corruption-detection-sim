from ..planners.base_planner import PlannerOutput

class BaseController:
    def step(self, plan: PlannerOutput, estimate):
        raise NotImplementedError("The step method should be overridden by a subclass.")
