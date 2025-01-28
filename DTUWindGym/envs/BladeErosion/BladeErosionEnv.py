import sys
import pickle
import os
import random
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque
from scipy.stats import expon, norm
from scipy.interpolate import interp1d, RegularGridInterpolator
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
import pandas as pd
import xarray as xr
import gymnasium as gym
import numpy as np

np.random.seed(None)

# UK: 2.0359939116469547e-08
# NO: 3.477390141887285e-08


class BladeErosionEnv:
    def __init__(self, fname, tbi=1800, Nt=1000, repair_cost=300_000, N_sim_yrs=5):
        # generate reference weather data
        data = (
            xr.open_dataset(fname)
            .sel(height=100)
            .to_dataframe()[["WS", "WD", "TKE", "PRECIP"]]
            .dropna()
        )
        # scale precipitation to mm/hr (raw data is in 30-min accumulated)
        data["PRECIP"] = data["PRECIP"] * 2
        # only use rain above 0.1 mm/hr and wind speed above 10 m/s
        self.data = data[((data["PRECIP"] > 0.1) & (data["WS"] > 12))]
        self.Nt = Nt
        self.tbi = tbi
        # run background weather data analysis
        self.repair_cost = repair_cost  # full rotor repair cost
        self.power_scaler = 39.79001338083535  # to compensate for the filtering
        # to compensate for accelerated episodes + vts**6.7 and irg**(2/3) scaling
        self.impingement_scaler = 1.0823127489000354e-08
        self.exponent = 3
        # spaces
        self.action_space = np.linspace(0, 1, 5)
        self.action_size = len(self.action_space)
        self.state_size = 3

        # power interpolator
        self.wt_power = RegularGridInterpolator(
            (
                np.concatenate(
                    (
                        np.arange(4, 6, 0.5),
                        np.arange(6, 12, 0.01),
                        np.arange(14, 25) + 1,
                    )
                ),
                np.linspace(0, 1, 51),
            ),
            np.genfromtxt("data/LUT10MW_P.csv", delimiter=","),
            bounds_error=False,
            fill_value=0,
        )

        # tip speed interpolator
        self.tip_speed_func = interp1d(
            np.arange(4, 26),
            89.15
            * np.array(
                [
                    6,
                    6,
                    6,
                    6,
                    6.423,
                    7.225,
                    8.029,
                    8.837,
                    9.6,
                    9.6,
                    9.6,
                    9.6,
                    9.6,
                    9.6,
                    9.6,
                    9.6,
                    9.6,
                    9.6,
                    9.6,
                    9.6,
                    9.6,
                    9.6,
                ]
            )
            * 2
            * np.pi
            / 60,
            bounds_error=False,
            fill_value=0,
        )

        # load damage model
        lut = pd.read_csv("data/LUT.csv")
        # compute CIs
        Di = norm.ppf(np.arange(1, 100) / 100).reshape(1, -1) * lut[
            "pred_sd"
        ].values.reshape(-1, 1) + lut["pred_mu"].values.reshape(-1, 1)
        Di[Di < 0] = 0
        Di[Di > 1] = 1
        self.Dfunc = RegularGridInterpolator(
            (lut["impingement"].values, np.arange(1, 100)),
            Di,
            bounds_error=False,
            fill_value=None,
        )

    # damage cost interpolator
    def _damage_cost_func(self, d):
        return self.repair_cost * d**self.exponent

    def no_damage_reward(self):
        EP = (
            self.power_scaler
            * self.tbi
            / 3600
            * self.wt_power(np.column_stack((self.ws, np.zeros(len((self.ws))))))
            * 1e-6
        )
        return np.sum(EP * self._price(self.ws))

    def reset(self, seed=None, percentile=50):
        # np.random.seed(seed)
        self.i = 0  # internal counter
        self.ni = 0  # acc. impingement
        self.percentile = percentile
        self.penalty = []
        self.EP = []
        self.episode_data = self.data.sample(
            self.Nt, replace=True, random_state=seed
        )  # sample from data
        self.ws = self.episode_data["WS"].values
        self.precip = self.episode_data["PRECIP"].values
        # wind speed, precipitation, damage
        return np.array([self.ws[self.i], self.precip[self.i], 0]).astype("float32")

    def step(self, action, no_repair_cost=False):
        ws = self.ws[self.i]
        precip = self.precip[self.i]
        ts0 = self.tip_speed_func(ws)  # reference tip speed [m/s]
        # get actual tip speed from action
        ts_min = self.tip_speed_func(4)
        ts = (ts0 - ts_min) * action + ts_min
        # compute initial damage
        d0 = self.Dfunc(np.array([self.ni, self.percentile]))[0]
        if ts0 == 0:
            EP = 0
        else:
            # energy production [MWh]
            EP = (
                self.tbi
                / 3600
                * self.wt_power(np.array([ws, d0]))[0]
                * (ts / ts0)
                * 1e-6
            )
            if no_repair_cost:
                # energy production [MWh]
                EP = (
                    self.tbi
                    / 3600
                    * self.wt_power(np.array([ws, 0]))[0]
                    * (ts / ts0)
                    * 1e-6
                )
        EP = EP * self.power_scaler  # scale energy production
        # compute acc. impingement
        # *self.rescale_param
        self.ni += self._calc_impingement(ws, precip, ts)[1]
        D = self.Dfunc(np.array([self.ni, self.percentile]))[0]
        # compute damage
        if no_repair_cost:
            D = 0
        # compute reward: reward energy production and penalize damage increment
        income_term = EP * self._price(ws)
        self.EP.append(EP)
        damage_term = self._damage_cost_func(D) - self._damage_cost_func(d0)
        self.penalty.append(damage_term)
        reward = income_term - damage_term
        self.i += 1
        new_state = np.array([self.ws[self.i], self.precip[self.i], D])
        done = False
        if (self.i + 1 == self.Nt) or (D == 1):
            done = True
        return new_state.astype("float32"), reward, done

    def _price(self, x):
        """Computes the electricity price given the wind speed (Could be made probabilistic)"""
        return np.polyval([-2.23453397e-02, -5.25233388e-01, 3.49885266e01], x)

    def _calc_impingement(self, ws, irg, vts):
        """Calculates the rain impingement"""
        # Rain drop diameter [mm] as a function of rain rate on the ground [mm/h]
        radi = np.array(
            [0.01, 0.8, 1.1, 1.25, 1.6, 1.9, 2.2, 2.6, 2.85, 3.05, 3.2, 3.5, 3.75]
        )
        rara = np.array([0, 0.1, 1, 2, 5, 10, 20, 40, 60, 80, 100, 150, 200])
        # interpolate rain rate to get rain drop diameter [mm]
        rdsb = np.interp(irg, rara, radi)
        vrg = (
            0.0481 * rdsb**3 - 0.8037 * rdsb**2 + 4.621 * rdsb
        )  # falling rain velocity [m/s]
        irg = irg / (1000 * 3600)  # convert rain rate from mm/hr to m/s
        vca = irg / vrg  # relative volume of water in the air [-]
        vrb = np.sqrt(ws**2 + vts**2)  # speed of rain hitting the blade [m/s]
        irb = vca * vrb  # rain rate on tip of blade [m/s]
        ni = irb * self.tbi  # rain column on the blade tip in one bin [m]
        ni = ni * (vts**6.7 * irg ** (2 / 3))
        return ni, ni * self.impingement_scaler
