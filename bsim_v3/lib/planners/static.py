from .base_planner import BasePlanner, PlannerOutput
from .utils import generate_figure_eight_approximation


class StaticFigureEightPlanner(BasePlanner):
    def __init__(self, center, length, width, num_points, target_velocity_fn):
        self.center = center
        self.length = length
        self.width = width
        self.num_points = num_points
        self.points, self.headings, self.curvatures, self.dK_ds_list = (
            generate_figure_eight_approximation(center, length, width, num_points)
        )
        self.velocities = target_velocity_fn(
            self.points, self.headings, self.curvatures, self.dK_ds_list
        )
        self.plan_id = "figure_eight"
        self.planner_output = PlannerOutput(
            self.plan_id,
            self.points,
            self.headings,
            self.curvatures,
            self.dK_ds_list,
            self.velocities,
        )

    def plan(self, estimated_state):
        return self.planner_output

    @property
    def base_plan(self):
        return self.planner_output