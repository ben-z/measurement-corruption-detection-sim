from collections import namedtuple

PlannerOutput = namedtuple("PlannerOutput", ["id", "points", "headings", "curvatures", "dK_ds_list", "velocities"])

class BasePlanner:
    def plan(self, estimated_state) -> PlannerOutput:
        raise NotImplementedError("The plan method should be overridden by a subclass.")
    
    @property
    def base_plan(self) -> PlannerOutput:
        """
        Returns the base path of the planner. This is the reference path prior to any
        modifications such as accounting for obstacles or other agents.
        """
        raise NotImplementedError("The base_points method should be overridden by a subclass.")