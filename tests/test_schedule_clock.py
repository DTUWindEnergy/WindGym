"""WindFarmEnv: a wd schedule's t = 0 is the first control step, not somewhere
inside reset's sensor warm-up (plan 2026-10-29 WP2; the trainer's in-run eval
scored agg-120 agents 600 s into the Richards turn because of this)."""
import numpy as np
from py_wake.examples.data.hornsrev1 import V80

from WindGym import WindFarmEnv
from test_utils import get_fast_pywake_config


def _step_schedule(t):
    return 270.0 + 10.0 * (t > 0)


def test_schedule_t0_is_the_first_control_step():
    d = V80().diameter()
    env = WindFarmEnv(turbine=V80(), x_pos=np.arange(2) * 5.0 * d, y_pos=np.zeros(2),
                      config=get_fast_pywake_config(), backend="dynamiks", turbtype="None",
                      reset_init=False, n_passthrough=0.5, burn_in_passthroughs=0.05,
                      fill_window=True, dt_sim=1, dt_env=2, wd_function=_step_schedule,
                      max_turb_move=1000.0)   # the frame slew never binds: dynamiks' wd_slow is a CENTRED
                                              # rate limit that would otherwise start a sharp step early
    try:
        env.reset(seed=0)
        assert env.steps_on_reset >= 2 and env.sim_steps_per_env_step == 2
        # every warm-up sample sat at the schedule's wd(0) = the base wd
        wd_hist = np.asarray(env.farm_measurements.turb_mes[0].wd.measurements, dtype=float)
        assert np.allclose(wd_hist, 270.0), wd_hist
        assert env.wd == 270.0
        env.step(np.zeros(2, dtype=np.float32))
        assert env.wd == 280.0          # the first control env step runs t > 0
    finally:
        env.close()
