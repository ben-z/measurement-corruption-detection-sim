from collections import namedtuple

PlannerOutput = namedtuple("PlannerOutput", ["id", "points", "headings", "curvatures", "dK_ds_list", "velocities"])

class BasePlanner:
    def plan(self, estimated_state) -> PlannerOutput:
        raise NotImplementedError("The plan method should be overridden by a subclass.")