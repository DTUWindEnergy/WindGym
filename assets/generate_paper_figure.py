"""Generate the flowfield figure for the JOSS paper."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patheffects as path_effects

from WindGym import FarmEval
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80

turbine = V80()
x_pos, y_pos = generate_square_grid(turbine, nx=3, ny=1, xDist=7, yDist=7)

config = {
    "yaw_init": "Zeros",
    "BaseController": "Local",
    "ActionMethod": "wind",
    "Track_power": False,
    "farm": {"yaw_min": -45, "yaw_max": 45},
    "wind": {
        "ws_min": 9,
        "ws_max": 9,
        "TI_min": 0.0,
        "TI_max": 0.0,
        "wd_min": 270,
        "wd_max": 270,
    },
    "act_pen": {"action_penalty": 0.0, "action_penalty_type": "Change"},
    "power_def": {"Power_reward": "Baseline", "Power_avg": 10, "Power_scaling": 1.0},
    "mes_level": {
        "turb_ws": True,
        "turb_wd": True,
        "turb_TI": False,
        "turb_power": False,
        "farm_ws": False,
        "farm_wd": False,
        "farm_TI": False,
        "farm_power": False,
    },
    "ws_mes": {
        "ws_current": True,
        "ws_rolling_mean": False,
        "ws_history_N": 1,
        "ws_history_length": 25,
        "ws_window_length": 25,
    },
    "wd_mes": {
        "wd_current": True,
        "wd_rolling_mean": False,
        "wd_history_N": 1,
        "wd_history_length": 20,
        "wd_window_length": 20,
    },
    "yaw_mes": {
        "yaw_current": True,
        "yaw_rolling_mean": False,
        "yaw_history_N": 1,
        "yaw_history_length": 10,
        "yaw_window_length": 10,
    },
    "power_mes": {
        "power_current": False,
        "power_rolling_mean": False,
        "power_history_N": 1,
        "power_history_length": 10,
        "power_window_length": 10,
    },
}

env = FarmEval(
    turbine=turbine,
    x_pos=x_pos,
    y_pos=y_pos,
    config=config,
    render_mode="rgb_array",
    seed=42,
)
env.set_wind_vals(ws=12.0, ti=0.04, wd=275.0)
obs, info = env.reset()

# Run steps with zero yaw offset so wakes develop clearly
for _ in range(20):
    action = np.zeros(env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)

# Get raw flow field data for custom plotting
ff = env.renderer.get_flow_field(env.fs, turbine=turbine)

fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")

mesh = ax.pcolormesh(
    ff["x"],
    ff["y"],
    np.asarray(ff["uvw"][0]).T,
    shading="nearest",
    cmap="viridis",
    vmin=5.0,
    vmax=12.0,
)
cbar = plt.colorbar(mesh, ax=ax, label="Wind Speed (m/s)")

# Draw turbines as ellipses
for ii, (x_, y_, r, yaw_, tilt_) in enumerate(
    zip(
        ff["x_turb"],
        ff["y_turb"],
        ff["diameter"] / 2,
        ff["yaw_plot"],
        ff["tilt"],
    )
):
    ax.add_artist(
        Ellipse(
            (x_, y_),
            2 * r * np.sin(np.deg2rad(tilt_)),
            2 * r,
            angle=yaw_,
            ec="k",
            fc="None",
        )
    )
    ax.plot(x_, y_, ".", color="k")
    text = ax.annotate(
        f"T {ii + 1}",
        (x_ - r, y_ + r * 1.75),
        fontsize=12,
        color="white",
    )
    text.set_path_effects(
        [
            path_effects.Stroke(linewidth=2, foreground="black"),
            path_effects.Normal(),
        ]
    )

ax.set_xlim(ff["x"].min(), 1650)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(f"Flow field at {ff['time']:.1f} s")

fig.savefig("assets/flowfield.png", dpi=300, bbox_inches="tight")
plt.close(fig)
env.close()
print("Saved assets/flowfield.png")
