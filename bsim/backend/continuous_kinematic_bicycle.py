import control
import numpy as np

def continuous_kinematic_bicycle_model(t, x, u, params):
    # Kinematic bicycle model (rear axle reference frame, continuous time)
    # x = [x, y, theta, v, delta]
    # u = [a, delta_dot]

    L = 2.9
    if 'L' in params:
        L = params['L']

    x_dot = np.zeros(5)
    x_dot[0] = x[3] * np.cos(x[2])
    x_dot[1] = x[3] * np.sin(x[2])
    x_dot[2] = x[3] * np.tan(x[4]) / L
    x_dot[3] = u[0]
    x_dot[4] = u[1]

    return x_dot

def make_continuous_kinematic_bicycle_model(L=2.9):
    return control.NonlinearIOSystem(
        continuous_kinematic_bicycle_model,
        inputs=('a', 'delta_dot'),
        outputs=('x', 'y', 'theta', 'v', 'delta'),
        states=('x', 'y', 'theta', 'v', 'delta'),
        params={'L': L},
    )

if __name__ == '__main__':
    sys = make_continuous_kinematic_bicycle_model()
    print(sys)
    pass
