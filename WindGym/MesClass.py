from collections import deque
import itertools
import numpy as np
from .WindEnv import WindEnv

"""
This file contains the classes for the measurements of the wind farm.

It is divided into three classes:
- Mes: Baseclass for the measurements
- turb_mes: Class for the measurements of the turbines
- farm_mes: Class for the measurements of the farm

The mes class is a baseclass for a type of measurements. It is just a deque with a get_measurements function that can return the desired measurements.

The turb_mes class is a class for the measurements of the turbines. It contains mes classes for the type of sensors we want to use. It also contains a function for calculating the turbulence intensity.

the farm_mes class is the senosrs for the farm. It contains a list of the turb_mes classes, and a mes class for the farm. It also contains a function for calculating the turbulence intensity for the farm.

"""


class Mes:
    """
    Baseclass for the measurements,
    we can decide how large a memory we need, and also how many measurements we want to get back
    Current: bool, if true return the latest measurement
    Rolling Mean: bool, if true return the rolling mean of the measurements
    history_N: int, number of rolling windows to use for the rolling mean. If 1, only return the latest value, if 2 return the lates and oldest value, if more then do some inbetween values also
    history_length: int, number of measurements to save
    window_length: int, size of the rolling window
    """

    def __init__(
        self,
        current=True,
        rolling_mean=False,
        history_N=3,
        history_length=100,
        window_length=5,
    ):
        self.current = current  # Do you want the current wind speed to be included in the measurements
        self.rolling_mean = rolling_mean  # Do you want the rolling mean of the wind speed to be included in the measurements
        self.history_N = (
            history_N  # Number of rolling windows to use for the rolling mean
        )
        self.history_length = history_length  # Number of measurements to save
        self.window_length = window_length  # Size of the rolling window

        self.measurements = deque(maxlen=self.history_length)

    def __call__(self, measurement):
        """
        Append the measurement to the deque via the call function
        """
        self.measurements.append(measurement)

    def append(self, measurement):
        """
        Append the measurement to the deque via the append function
        """
        self.measurements.append(measurement)

    def add_measurement(self, measurement):
        """Append the measurement to the deque"""
        self.measurements.append(measurement)

    def get_measurements(self):
        """
        Get the desired measurements with graceful handling of startup period
        """
        return_vals = []
        if len(self.measurements) == 0:
            return np.array(return_vals, dtype=np.float32)

        if self.current:
            # Return the current measurement
            return_vals.append(np.mean(self.measurements[-1]))

        if self.rolling_mean:
            available_length = len(self.measurements)

            for i in range(self.history_N):
                if i == 0:
                    # Latest window
                    start = max(0, available_length - self.window_length)
                    result = list(
                        itertools.islice(self.measurements, start, available_length)
                    )

                elif i == self.history_N - 1 and available_length >= self.window_length:
                    # Oldest window (only if we have enough data)
                    result = list(
                        itertools.islice(self.measurements, 0, self.window_length)
                    )

                else:
                    # Middle windows - space them out based on available data
                    if available_length < self.window_length:
                        # If we don't have enough data, use all available data
                        result = list(self.measurements)
                    else:
                        # Calculate position for this window
                        spacing = max(
                            1,
                            (available_length - self.window_length)
                            // (self.history_N - 1),
                        )
                        pos = min(i * spacing, available_length - self.window_length)
                        result = list(
                            itertools.islice(
                                self.measurements, pos, pos + self.window_length
                            )
                        )

                # Always calculate mean of whatever data we have
                # if result:
                return_vals.append(np.mean(result))
                # else:
                #    # If somehow we got no data, use the latest value
                #    return_vals.append(np.mean(self.measurements[-1]))

        return np.array(return_vals, dtype=np.float32)


class turb_mes:
    """
    Class for all measurements.
    Each turbine stores measurements for wind speed, wind direction and yaw angle...
    """

    def __init__(
        self,
        ws_current=False,
        ws_rolling_mean=True,
        ws_history_N=1,
        ws_history_length=10,
        ws_window_length=10,
        wd_current=False,
        wd_rolling_mean=True,
        wd_history_N=1,
        wd_history_length=10,
        wd_window_length=10,
        yaw_current=False,
        yaw_rolling_mean=True,
        yaw_history_N=2,
        yaw_history_length=30,
        yaw_window_length=1,
        power_current=False,
        power_rolling_mean=True,
        power_history_N=1,
        power_history_length=10,
        power_window_length=10,
        ws_min: float = 7.0,
        ws_max: float = 20.0,
        wd_min: float = 270.0,
        wd_max: float = 360.0,
        yaw_min=-45,
        yaw_max=45,
        TI_min=0.00,
        TI_max=0.50,
        include_TI=True,
        power_max=2000000,  # 2 MW
    ):
        self.ws = Mes(
            current=ws_current,
            rolling_mean=ws_rolling_mean,
            history_N=ws_history_N,
            history_length=ws_history_length,
            window_length=ws_window_length,
        )
        self.wd = Mes(
            current=wd_current,
            rolling_mean=wd_rolling_mean,
            history_N=wd_history_N,
            history_length=wd_history_length,
            window_length=wd_window_length,
        )
        self.yaw = Mes(
            current=yaw_current,
            rolling_mean=yaw_rolling_mean,
            history_N=yaw_history_N,
            history_length=yaw_history_length,
            window_length=yaw_window_length,
        )
        self.power = Mes(
            current=power_current,
            rolling_mean=power_rolling_mean,
            history_N=power_history_N,
            history_length=power_history_length,
            window_length=power_window_length,
        )

        self.ws_max = ws_max
        self.ws_min = ws_min
        self.wd_min = wd_min
        self.wd_max = wd_max
        self.yaw_min = yaw_min
        self.yaw_max = yaw_max
        self.TI_min = TI_min
        self.TI_max = TI_max
        self.include_TI = include_TI
        self.power_max = power_max

        if self.include_TI:
            # If we want to include the TI, then we set the get_TI function to the calc_TI function
            self.get_TI = self.calc_TI
        else:
            # If not, then we just add an empty array. This is skipped in the np.concatenate function, so it should not matter
            self.get_TI = self.empty_np

    def empty_np(self, scaled=False):
        """
        Return an empty array
        """
        return np.array([])

    def calc_TI(self, scaled=False):
        """
        Calcualte TI from the wind speed measurements
        """

        u = np.array(
            self.ws.measurements
        )  # First we get the wind speed measurements and turn them into an array
        U = u.mean()  # Then we calculate the mean wind speed

        TI = np.array([np.std(u - U) / U], dtype=np.float32)  # Then we calculate the TI

        if scaled:
            # Scale the measurements
            return self._scale_val(TI, self.TI_min, self.TI_max)
        else:
            return TI

    def max_hist(self):
        """
        Return the maximum history length of the measurements
        """

        return max(
            [self.ws.history_length, self.wd.history_length, self.yaw.history_length]
        )

    def observed_variables(self):
        """
        Returns the number of observed variables
        """

        obs_var = 0

        # Number of ws variables
        obs_var += self.ws.current + self.ws.rolling_mean * self.ws.history_N

        # Number of wd variables
        obs_var += self.wd.current + self.wd.rolling_mean * self.wd.history_N

        # Number of yaw variables
        obs_var += self.yaw.current + self.yaw.rolling_mean * self.yaw.history_N

        # Number of TI variables
        obs_var += self.include_TI

        # Number of power variables
        obs_var += self.power.current + self.power.rolling_mean * self.power.history_N

        return obs_var

    def add_ws(self, measurement):
        # add measurements to the ws
        self.ws.add_measurement(measurement)

    def add_wd(self, measurement):
        # add measurements to the wd
        self.wd.add_measurement(measurement)

    def add_yaw(self, measurement):
        # add measurements to the yaw
        self.yaw.add_measurement(measurement)

    def add_power(self, measurement):
        # add measurements to the power
        self.power.add_measurement(measurement)

    def get_ws(self, scaled=False):
        # get the ws measurements
        if scaled:
            # Scale the measurements
            return self._scale_val(self.ws.get_measurements(), self.ws_min, self.ws_max)
        else:
            return self.ws.get_measurements()

    def get_wd(self, scaled=False):
        # get the wd measurements
        if scaled:
            # Scale the measurements
            return self._scale_val(self.wd.get_measurements(), self.wd_min, self.wd_max)
        else:
            return self.wd.get_measurements()

    def get_yaw(self, scaled=False):
        # get the yaw measurements
        if scaled:
            # Scale the measurements
            return self._scale_val(
                self.yaw.get_measurements(), self.yaw_min, self.yaw_max
            )
        else:
            return self.yaw.get_measurements()

    def get_power(self, scaled=False):
        # get the power measurements
        if scaled:
            # Scale the measurements
            return self._scale_val(
                self.power.get_measurements(), 0, self.power_max
            )  # Min power is 0
        else:
            return self.power.get_measurements()

    def _scale_val(self, val, min_val, max_val):
        # Scale the value from -1 to 1
        return 2 * (val - min_val) / (max_val - min_val) - 1

    def get_measurements(self, scaled=False):
        # get all the measurements
        if scaled:
            # Scale the measurements
            return np.concatenate(
                [
                    self._scale_val(self.get_ws(), self.ws_min, self.ws_max),
                    self._scale_val(self.get_wd(), self.wd_min, self.wd_max),
                    self._scale_val(self.get_yaw(), self.yaw_min, self.yaw_max),
                    self._scale_val(self.get_TI(), self.TI_min, self.TI_max),
                    self._scale_val(self.get_power(), 0, self.power_max),
                ]
            )
        else:
            # Return the measurements
            return np.concatenate(
                [
                    self.get_ws(),
                    self.get_wd(),
                    self.get_yaw(),
                    self.get_TI(),
                    self.get_power(),
                ]
            )


class farm_mes(WindEnv):
    """
    Class for the measurements of the farm.
    The farm stores measurements from each turbine for wind speed, wind direction, yaw angle, power
    """

    def __init__(
        self,
        n_turbines,
        noise="None",  # Can be: "None", or "Normal"   Ill add OU later
        turb_ws=True,
        turb_wd=True,
        turb_TI=True,
        turb_power=True,
        farm_ws=True,
        farm_wd=True,
        farm_TI=False,
        farm_power=False,
        ws_current=False,
        ws_rolling_mean=True,
        ws_history_N=1,
        ws_history_length=10,
        ws_window_length=10,
        wd_current=False,
        wd_rolling_mean=True,
        wd_history_N=1,
        wd_history_length=10,
        wd_window_length=10,
        yaw_current=False,
        yaw_rolling_mean=True,
        yaw_history_N=2,
        yaw_history_length=30,
        yaw_window_length=1,
        power_current=False,
        power_rolling_mean=True,
        power_history_N=1,
        power_history_length=10,
        power_window_length=10,
        ws_min: float = 7.0,
        ws_max: float = 20.0,
        wd_min: float = 270.0,
        wd_max: float = 360.0,
        yaw_min=-45,
        yaw_max=45,
        TI_min=0.00,
        TI_max=0.50,
        power_max=2000000,  # 2 MW
    ):
        self.n_turbines = n_turbines
        self.turb_mes = []
        self.turb_ws = turb_ws  # do we want measurements from the turbines individually
        self.turb_wd = turb_wd  # do we want measurements from the turbines individually
        self.turb_TI = turb_TI  # do we want measurements from the turbines individually
        self.turb_power = (
            turb_power  # do we want measurements from the turbines individually
        )

        self.farm_ws = farm_ws  # do we want measurements from the farm, i.e. the average of the turbines
        self.farm_wd = farm_wd  # do we want measurements from the farm, i.e. the average of the turbines
        self.farm_TI = farm_TI
        self.farm_power = farm_power

        self.ws_max = ws_max
        self.ws_min = ws_min
        self.wd_min = wd_min
        self.wd_max = wd_max
        self.yaw_min = yaw_min
        self.yaw_max = yaw_max
        self.power_max = power_max

        if turb_TI or farm_TI:
            # If we return the TI, then we check for the number of data points, and then set the TI_fun to the get_TI function
            if (
                ws_history_length < 60
            ):  # TODO figure out the best number of data points to use
                print(
                    f"Warning: You are only saving the last {ws_history_length} data measurements of wind speed. It is these data points that are used for the TI calculations, so you might need more data points to get a good estimate of the TI. "
                )

        # TODO Add Ornstein–Uhlenbeck process for noise
        self.ws_noise = 0.0
        self.wd_noise = 2.0  # 2 degrees
        self.yaw_noise = 0.0
        self.power_noise = 0.0

        if noise == "Normal":
            self._add_noise = self._random_normal
        elif noise == "None":
            self._add_noise = self._return_zeros

        # We assume that the farm oberservations would take the same form as the turbine observations.
        self.farm_observed_variables = (
            farm_ws * (ws_current + ws_rolling_mean * ws_history_N)
            + farm_wd * (wd_current + wd_rolling_mean * wd_history_N)
            + farm_TI
            + farm_power * (power_current + power_rolling_mean * power_history_N)
        )

        # For each turbine, create a class of turbine measurements
        for _ in range(n_turbines):
            self.turb_mes.append(
                turb_mes(
                    ws_current and turb_ws,
                    ws_rolling_mean and turb_ws,
                    ws_history_N,
                    ws_history_length,
                    ws_window_length,
                    wd_current and turb_wd,
                    wd_rolling_mean and turb_wd,
                    wd_history_N,
                    wd_history_length,
                    wd_window_length,
                    yaw_current,
                    yaw_rolling_mean,
                    yaw_history_N,
                    yaw_history_length,
                    yaw_window_length,
                    power_current and turb_power,
                    power_rolling_mean and turb_power,
                    power_history_N,
                    power_history_length,
                    power_window_length,
                    self.ws_min,
                    self.ws_max,
                    wd_min,
                    wd_max,
                    yaw_min,
                    yaw_max,
                    TI_min,
                    TI_max,
                    turb_TI,
                    power_max,
                )
            )

        # Create a class for the farm measurements. This is still used for the info dictionary, so even if there are no farm level measurements returned, it is still created
        self.farm_mes = turb_mes(
            ws_current and farm_ws,
            ws_rolling_mean and farm_ws,
            ws_history_N,
            ws_history_length,
            ws_window_length,
            wd_current and farm_wd,
            wd_rolling_mean and farm_wd,
            wd_history_N,
            wd_history_length,
            wd_window_length,
            yaw_current,
            yaw_rolling_mean,
            yaw_history_N,
            yaw_history_length,
            yaw_window_length,
            power_current and farm_power,
            power_rolling_mean and farm_power,
            power_history_N,
            power_history_length,
            power_window_length,
            self.ws_min,
            self.ws_max,
            wd_min,
            wd_max,
            yaw_min,
            yaw_max,
            TI_min,
            TI_max,
            farm_TI,
            power_max=power_max * n_turbines,
        )  # The max power is the sum of all the turbines

        if self.farm_TI:
            # If we want to include the TI, then we set the get_TI function to the calc_TI function
            self.get_TI = self.get_TI_farm
        else:
            # If not, then we just add an empty array. This is skipped in the np.concatenate function, so it should not matter
            self.get_TI = self.empty_np

    def empty_np(self, scaled=False):
        # Return an empty array
        return np.array([])

    def add_ws(self, measurement):
        measurement += self._add_noise(mean=0, std=self.ws_noise, n=len(measurement))
        # add measurements to the ws
        for turb in self.turb_mes:
            turb.add_ws(measurement)
        if self.farm_ws:
            self.farm_mes.add_ws(np.mean(measurement))

    def add_wd(self, measurement):
        # add measurements to the wd
        measurement += self._add_noise(mean=0, std=self.wd_noise, n=len(measurement))
        for turb in self.turb_mes:
            turb.add_wd(measurement)
        if self.farm_wd:
            self.farm_mes.add_wd(np.mean(measurement))

    def add_yaw(self, measurement):
        # add measurements to the yaw
        measurement += self._add_noise(mean=0, std=self.yaw_noise, n=len(measurement))
        for turb in self.turb_mes:
            turb.add_yaw(measurement)

    def add_power(self, measurement):
        # add measurements to the power
        measurement += self._add_noise(mean=0, std=self.power_noise, n=len(measurement))
        for turb in self.turb_mes:
            turb.add_power(measurement)
        if self.farm_power:
            self.farm_mes.add_power(
                np.sum(measurement)
            )  # Instead of mean, we sum the power

    def add_measurements(self, ws, wd, yaws, powers):
        # add measurements to the ws, wd and yaw in one go

        # Adding noise to the measurements
        # print("Before noise in farm: ws: ", ws, "| wd: ", wd, "| yaws: ", yaws, "| powers: ", powers)

        ws += self._add_noise(mean=0, std=self.ws_noise, n=len(ws))
        wd += self._add_noise(mean=0, std=self.wd_noise, n=len(wd))
        yaws += self._add_noise(mean=0, std=self.yaw_noise, n=len(yaws))
        powers += self._add_noise(mean=0, std=self.power_noise, n=len(powers))
        # print("After norise in farm: ws: ", ws, "| wd: ", wd, "| yaws: ", yaws, "| powers: ", powers)

        for turb, speed, direction, yaw, power in zip(
            self.turb_mes, ws, wd, yaws, powers
        ):
            turb.add_ws(speed)
            turb.add_wd(direction)
            turb.add_yaw(yaw)
            turb.add_power(power)

        # Add to farm level measurements
        self.farm_mes.add_ws(np.mean(ws))
        self.farm_mes.add_wd(np.mean(wd))
        self.farm_mes.add_power(np.sum(powers))

        #
        # if self.farm_ws:
        #     self.farm_mes.add_ws(np.mean(ws))

        # if self.farm_wd:
        #     self.farm_mes.add_wd(np.mean(wd))

        # if self.farm_power:
        #     self.farm_mes.add_power(np.sum(powers))

    def max_hist(self):
        """
        Return the maximum history length of the measurements
        """

        return self.turb_mes[0].max_hist()

    def observed_variables(self):
        """
        Returns the number of observed variables
        """

        return (
            self.turb_mes[0].observed_variables() * self.n_turbines
            + self.farm_observed_variables
        )

    def get_ws_turb(self, scaled=False):
        # get the ws measurements
        return np.array(
            [turb.get_ws(scaled=scaled) for turb in self.turb_mes]
        ).flatten()

    def get_ws_farm(self, scaled=False):
        # get the ws measurements
        if scaled:
            # Scale the measurements
            return self._scale_val(self.farm_mes.get_ws(), self.ws_min, self.ws_max)
        else:
            return self.farm_mes.get_ws()

    def get_power_turb(self, scaled=False):
        # get the power measurements
        return np.array(
            [turb.get_power(scaled=scaled) for turb in self.turb_mes]
        ).flatten()

    def get_power_farm(self, scaled=False):
        # get the power measurements
        if scaled:
            # Scale the measurements
            return self._scale_val(
                self.farm_mes.get_power(), 0, self.power_max * self.n_turbines
            )  # Min power is 0, max is the sum of all the turbines
        else:
            return self.farm_mes.get_power()

    def get_wd_turb(self, scaled=False):
        # get the wd measurements
        return np.array(
            [turb.get_wd(scaled=scaled) for turb in self.turb_mes]
        ).flatten()

    def get_wd_farm(self, scaled=False):
        # get the wd measurements
        if scaled:
            # Scale the measurements
            return self._scale_val(self.farm_mes.get_wd(), self.wd_min, self.wd_max)
        else:
            return self.farm_mes.get_wd()

    def get_TI_turb(self, scaled=False):
        # get the TI measurements
        return np.array(
            [turb.calc_TI(scaled=scaled) for turb in self.turb_mes]
        ).flatten()

    def get_TI_farm(self, scaled=False):
        # Return the average value of the TI measurements
        TI_farm = self.get_TI_turb(scaled=scaled)
        return TI_farm.mean()

    def get_yaw_turb(self, scaled=False):
        # get the yaw measurements
        return np.array([turb.get_yaw(scaled) for turb in self.turb_mes]).flatten()

    def get_measurements(self, scaled=False):
        # get all the measurements
        # if scaled is true, then the measurements are scaled between -1 and 1
        farm_measurements = np.array([])

        if self.farm_ws:
            ws_farm = self.get_ws_farm(scaled=scaled)
            farm_measurements = np.append(farm_measurements, ws_farm)

        if self.farm_wd:
            wd_farm = self.get_wd_farm(scaled=scaled)
            farm_measurements = np.append(farm_measurements, wd_farm)

        if self.farm_TI:
            TI_farm = self.get_TI(scaled=scaled)
            farm_measurements = np.append(farm_measurements, TI_farm)

        if self.farm_power:
            power_farm = self.get_power_farm(scaled=scaled)
            farm_measurements = np.append(farm_measurements, power_farm)

        turb_measurements = np.array(
            [turb.get_measurements(scaled=scaled) for turb in self.turb_mes]
        ).flatten()

        return np.concatenate([turb_measurements, farm_measurements])
