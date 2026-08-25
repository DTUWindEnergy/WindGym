from .wind_farm_env import WindFarmEnv


"""
This is a wrapper for the WindFarmEnv class. It is used to evaluate the environment with specific wind values.
The difference is that we can set the wind values directly, and we can also set the yaw values directly.
"""


class FarmEval(WindFarmEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        turbine,
        x_pos,
        y_pos,
        finite_episode: bool = False,
        # Max and min values for the turbulence intensity measurements. Used for internal scaling
        ws_scaling_min: float = 0.0,
        ws_scaling_max: float = 30.0,
        wd_scaling_min: float = 0,
        wd_scaling_max: float = 360,
        ti_scaling_min: float = 0.0,
        ti_scaling_max: float = 1.0,
        yaw_scaling_min: float = -45,
        yaw_scaling_max: float = 45,
        yaw_init="Zeros",
        TurbBox="Default",
        config=None,
        power_ref_function=None,
        Baseline_comp=False,
        render_mode=None,
        turbtype="MannGenerate",
        backend: str = "dynamiks",
        seed=None,
        dt_sim=1,  # Simulation timestep in seconds
        dt_env=1,  # Environment timestep in seconds
        yaw_step_sim=1,  # Environment timestep in seconds
        yaw_step_env=None,
        n_passthrough=5,
        HTC_path=None,
        reset_init=True,
        fill_window=True,
        sample_site=None,
        burn_in_passthroughs=2,
        tilt=None,
        cleanup_on_time_limit: bool = True,
        keep_hawc_results: bool = False,
        op_lookup=None,
        max_time_steps=None,
    ):
        self.finite_episode = finite_episode
        # TODO There must be a better way to set all these valuesm **kwargs???
        # Run the Env with these values, to make sure that the oberservartion space is the same.
        super().__init__(
            turbine=turbine,
            x_pos=x_pos,
            y_pos=y_pos,
            n_passthrough=n_passthrough,
            ws_scaling_min=ws_scaling_min,
            ws_scaling_max=ws_scaling_max,
            wd_scaling_min=wd_scaling_min,
            wd_scaling_max=wd_scaling_max,
            ti_scaling_min=ti_scaling_min,
            ti_scaling_max=ti_scaling_max,
            yaw_scaling_min=yaw_scaling_min,
            yaw_scaling_max=yaw_scaling_max,
            burn_in_passthroughs=burn_in_passthroughs,
            TurbBox=TurbBox,
            turbtype=turbtype,
            backend=backend,
            config=config,
            power_ref_function=power_ref_function,
            Baseline_comp=Baseline_comp,  # UPDATE: Changed so that we dont need the baseline farm anymore. Before it was always true! #We always want to compare to the baseline, so this is true
            yaw_init=yaw_init,
            render_mode=render_mode,
            seed=seed,
            dt_sim=dt_sim,  # Simulation timestep in seconds
            dt_env=dt_env,  # Environment timestep in seconds
            yaw_step_sim=yaw_step_sim,
            yaw_step_env=yaw_step_env,
            HTC_path=HTC_path,
            reset_init=reset_init,
            fill_window=fill_window,
            sample_site=sample_site,
            tilt=tilt,
            cleanup_on_time_limit=cleanup_on_time_limit,
            keep_hawc_results=keep_hawc_results,
            op_lookup=op_lookup,
            # Sized episode budget; needed by turbtype="Precursor" so the
            # random-window check sees the harness horizon rather than the
            # passthrough-derived time_max (finite_episode=False still lifts
            # time_max to "infinite" after reset).
            max_time_steps=max_time_steps,
        )
        self.yaml_path = config  # Saved for legacy reasons

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
            # Update wind_manager to use exact values
            self.wind_manager.ws_min = ws
            self.wind_manager.ws_max = ws
        if ti is not None:
            self.ti = ti
            self.TI_inflow_min = ti
            self.TI_inflow_max = ti
            # Update wind_manager to use exact values
            self.wind_manager.ti_min = ti
            self.wind_manager.ti_max = ti
        if wd is not None:
            self.wd = wd
            self.wd_inflow_min = wd
            self.wd_inflow_max = wd
            # Update wind_manager to use exact values
            self.wind_manager.wd_min = wd
            self.wind_manager.wd_max = wd
        if veer is not None:
            self.veer = veer
            self.veer_inflow_min = veer
            self.veer_inflow_max = veer
            # Degenerate interval -> deterministic value, no RNG draw
            self.wind_manager.veer_min = veer
            self.wind_manager.veer_max = veer

    def set_yaw_vals(self, yaw_vals):
        """
        Set the yaw values to be used in the evaluation
        """
        self.yaw_initial = yaw_vals

    def update_tf(self, path):
        """
        Overwrite the _def_site method to set the turbulence field to the path given
        """
        self.TF_files = [path]
