from .wind_farm_env import WindFarmEnv


class FarmEval(WindFarmEnv):
    """``WindFarmEnv`` for evaluation under fixed, user-set conditions.

    Differences from the base env:

    * ``set_wind_vals`` pins the wind speed / TI / direction / veer that the
      next ``reset`` samples (the sampling draw still happens, so seeded
      evaluations keep the base env's RNG stream).
    * ``set_yaw_vals`` sets the initial yaws (used with ``yaw_init="Defined"``).
    * ``update_tf`` pins a single turbulence box file (``turbtype="MannLoad"``).
    * Episodes never truncate unless ``finite_episode=True``: after ``reset``
      ``time_max`` is lifted to a large sandbox value.

    Defaults differ from the base env: ``turbtype="MannGenerate"`` and
    ``yaw_init="Zeros"``. Every other ``WindFarmEnv`` keyword argument is
    forwarded unchanged.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, turbine, x_pos, y_pos, finite_episode: bool = False, **kwargs):
        # Read by reset(), which the base __init__ calls when reset_init=True.
        self.finite_episode = finite_episode
        kwargs.setdefault("turbtype", "MannGenerate")
        kwargs.setdefault("yaw_init", "Zeros")
        super().__init__(turbine=turbine, x_pos=x_pos, y_pos=y_pos, **kwargs)
        self.yaml_path = kwargs.get("config")  # Saved for legacy reasons (raw argument)
        # So type(env)(**env.kwargs) clones the evaluation env faithfully.
        self.kwargs["finite_episode"] = finite_episode

    def reset(self, seed=None, options=None):
        # Overwrite the reset function so that we never terminates.
        observation, info = super().reset(seed=seed, options=options)
        # Only set a large "sandbox" time_max if the finite_episode flag is False.
        if not self.finite_episode:
            # Large enough that fixed-step evaluations never truncate (real eval
            # horizons are ~10^3 steps), but not absurdly so: eval's memory-
            # cleanup step sets timestep = time_max and takes one throwaway
            # env.step, which for a callable power_ref_function lazily extends
            # the reference trajectory up to that index. 9999999 made that a ~10M
            # entry blow-up; 100_000 keeps it cheap. PowerTrackingManager also
            # caps the extension defensively (MAX_TRAJECTORY_STEPS).
            self.time_max = 100_000

        return observation, info

    def set_wind_vals(self, ws=None, ti=None, wd=None, veer=None):
        """
        Set the wind values to be used in the evaluation.
        veer is the linear veer rate in deg per 100 m (0 at hub height).
        """
        if ws is not None:
            self.ws = ws
            self.ws_inflow_min = ws
            self.ws_inflow_max = ws
        if ti is not None:
            self.ti = ti
            self.TI_inflow_min = ti
            self.TI_inflow_max = ti
        if wd is not None:
            self.wd = wd
            self.wd_inflow_min = wd
            self.wd_inflow_max = wd
        if veer is not None:
            self.veer = veer
            self.veer_inflow_min = veer
            self.veer_inflow_max = veer
        # Update wind_manager to use exact values
        self.wind_manager.fix_conditions(ws=ws, ti=ti, wd=wd, veer=veer)

    def set_yaw_vals(self, yaw_vals):
        """
        Set the yaw values to be used in the evaluation
        """
        self.yaw_initial = yaw_vals

    def update_tf(self, path):
        """
        Pin the turbulence box: the next reset loads ``path`` (turbtype="MannLoad").
        """
        self.TF_files = [path]
