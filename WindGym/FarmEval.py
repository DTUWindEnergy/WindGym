from .Wind_Farm_Env import WindFarmEnv


"""
This is a wrapper for the WindFarmEnv class. It is used to evaluate the environment with specific wind values.
The difference is that we can set the wind values directly, and we can also set the yaw values directly.
"""


class FarmEval(WindFarmEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        turbine,
        # Max and min values for the turbulence intensity measurements. Used for internal scaling
        TI_min_mes: float = 0.0,
        TI_max_mes: float = 0.50,
        yaw_init="Zeros",
        TurbBox="Default",
        yaml_path=None,
        Baseline_comp=False,
        render_mode=None,
        turbtype="MannLoad",
        seed=None,
        dt_sim=1,  # Simulation timestep in seconds
        dt_env=1,  # Environment timestep in seconds
        yaw_step=1,  # Environment timestep in seconds
        n_passthrough=5,
        HTC_path=None,
        reset_init=True,
    ):
        # TODO There must be a better way to set all these valuesm **kwargs???
        # Run the Env with these values, to make sure that the oberservartion space is the same.
        super().__init__(
            turbine=turbine,
            n_passthrough=n_passthrough,
            TI_min_mes=TI_min_mes,
            TI_max_mes=TI_max_mes,
            TurbBox=TurbBox,
            turbtype=turbtype,
            yaml_path=yaml_path,
            Baseline_comp=Baseline_comp,  # UPDATE: Changed so that we dont need the baseline farm anymore. Before it was always true! #We always want to compare to the baseline, so this is true
            yaw_init=yaw_init,
            render_mode=render_mode,
            seed=seed,
            dt_sim=dt_sim,  # Simulation timestep in seconds
            dt_env=dt_env,  # Environment timestep in seconds
            yaw_step=yaw_step,
            HTC_path=HTC_path,
            reset_init=reset_init,
        )

    def reset(self, seed=None, options=None):
        # Overwrite the reset function so that we never terminates.
        # observation, info = WindFarmEnv.reset(self, seed, options)
        observation, info = super().reset(seed=seed, options=options)
        # Just set to a very high number, so that we never terminate.
        self.time_max = 9999999

        return observation, info

    def set_wind_vals(self, ws=None, ti=None, wd=None):
        """
        Set the wind values to be used in the evaluation
        """
        if ws is not None:
            self.ws = ws
            self.ws_min = ws
            self.ws_max = ws
        if ti is not None:
            self.ti = ti
            self.TI_min = ti
            self.TI_max = ti
        if wd is not None:
            self.wd = wd
            self.wd_min = wd
            self.wd_max = wd

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
