from __future__ import annotations
from collections import deque
from typing import Callable
import itertools
import numpy as np
from numpy.typing import NDArray
from .. import utils

"""
This file contains the classes for the measurements of the wind farm.

It is divided into three classes:
- Mes: Baseclass for the measurements
- TurbMes: Class for the measurements of the turbines
- FarmMes: Class for the measurements of the farm

The Mes class is a baseclass for a type of measurements. It is just a deque with a get_measurements function that can return the desired measurements.

The TurbMes class is a class for the measurements of the turbines. It contains Mes classes for the type of sensors we want to use. It also contains a function for calculating the turbulence intensity.

The FarmMes class is the sensors for the farm. It contains a list of the TurbMes classes, and a Mes class for the farm. It also contains a function for calculating the turbulence intensity for the farm.

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
        current: bool = True,
        rolling_mean: bool = False,
        history_N: int = 3,
        history_length: int = 100,
        window_length: int = 5,
    ) -> None:
        # Do you want the current wind speed to be included in the measurements
        self.current: bool = current
        # Do you want the rolling mean of the wind speed to be included in the measurements
        self.rolling_mean: bool = rolling_mean
        self.history_N: int = (
            history_N  # Number of rolling windows to use for the rolling mean
        )
        self.history_length: int = history_length  # Number of measurements to save
        self.window_length: int = window_length  # Size of the rolling window

        self.measurements: deque = deque(maxlen=self.history_length)

    def __call__(self, measurement: float | NDArray[np.floating]) -> None:
        """
        Append the measurement to the deque via the call function
        """
        self.measurements.append(measurement)

    def append(self, measurement: float | NDArray[np.floating]) -> None:
        """
        Append the measurement to the deque via the append function
        """
        self.measurements.append(measurement)

    def add_measurement(self, measurement: float | NDArray[np.floating]) -> None:
        """Append the measurement to the deque"""
        self.measurements.append(measurement)

    def get_measurements(self) -> NDArray[np.float32]:
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


class TurbMes:
    """
    Class for all measurements.
    Each turbine stores measurements for wind speed, wind direction and yaw angle...
    """

    def __init__(
        self,
        ws_current: bool = False,
        ws_rolling_mean: bool = True,
        ws_history_N: int = 1,
        ws_history_length: int = 10,
        ws_window_length: int = 10,
        wd_current: bool = False,
        wd_rolling_mean: bool = True,
        wd_history_N: int = 1,
        wd_history_length: int = 10,
        wd_window_length: int = 10,
        yaw_current: bool = False,
        yaw_rolling_mean: bool = True,
        yaw_history_N: int = 2,
        yaw_history_length: int = 30,
        yaw_window_length: int = 1,
        power_current: bool = False,
        power_rolling_mean: bool = True,
        power_history_N: int = 1,
        power_history_length: int = 10,
        power_window_length: int = 10,
        ws_min: float = 7.0,
        ws_max: float = 20.0,
        wd_min: float = 270.0,
        wd_max: float = 360.0,
        yaw_min: float = -45,
        yaw_max: float = 45,
        TI_min: float = 0.00,
        TI_max: float = 0.50,
        include_TI: bool = True,
        power_max: float = 2000000,  # 2 MW
        n_probes_per_turb: dict = {},
        ti_sample_count: int = 30,
    ) -> None:
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

        self.ws_hf_buffer: deque = deque(maxlen=ti_sample_count)
        self.ws_max: float = ws_max
        self.ws_min: float = ws_min
        self.wd_min: float = wd_min
        self.wd_max: float = wd_max
        self.yaw_min: float = yaw_min
        self.yaw_max: float = yaw_max
        self.TI_min: float = TI_min
        self.TI_max: float = TI_max
        self.include_TI: bool = include_TI
        self.power_max: float = power_max
        self.n_probes_per_turb: dict = n_probes_per_turb

        if self.include_TI:
            # Set get_TI function to calc_TI for TI calculations
            self.get_TI: Callable[[bool], NDArray[np.float32]] = self.calc_TI
        else:
            # Return empty array (skipped in np.concatenate)
            self.get_TI: Callable[[bool], NDArray[np.float32]] = self.empty_np

    def add_hf_ws(self, measurement: float) -> None:
        """Appends a single wind speed measurement to the high-frequency buffer."""
        self.ws_hf_buffer.append(measurement)

    def empty_np(self, scaled: bool = False) -> NDArray[np.float32]:
        """
        Return an empty array
        """
        return np.array([])

    def calc_TI(self, scaled: bool = False) -> NDArray[np.float32]:
        """
        Calcualte TI from the wind speed measurements
        """

        if len(self.ws_hf_buffer) < 2:
            return np.array([0.0], dtype=np.float32)

        u = np.array(
            self.ws_hf_buffer
        )  # Get measurements from the high-frequency buffer
        U = u.mean()  # Then we calculate the mean wind speed

        if U == 0:
            return np.array([0.0], dtype=np.float32)

        # Then we calculate the TI
        TI = np.array([np.std(u) / U], dtype=np.float32)

        if scaled:
            # Scale the measurements
            return self._scale_val(TI, self.TI_min, self.TI_max)
        else:
            return TI

    def max_hist(self) -> int:
        """
        Return the maximum history length of the measurements
        """

        return max(
            [
                self.ws.history_length,
                self.wd.history_length,
                self.yaw.history_length,
                self.power.history_length,
            ]
        )

    def observed_variables(self) -> int:
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

        # Number of probes per turbine
        if (
            hasattr(self, "n_probes_per_turb")
            and self.n_probes_per_turb is not None
            and 0 in self.n_probes_per_turb
        ):
            obs_var += self.n_probes_per_turb[0]

        return obs_var

    def add_ws(self, measurement: float | NDArray[np.floating]) -> None:
        # add measurements to the ws
        self.ws.add_measurement(measurement)

    def add_wd(self, measurement: float | NDArray[np.floating]) -> None:
        # add measurements to the wd
        self.wd.add_measurement(measurement)

    def add_yaw(self, measurement: float | NDArray[np.floating]) -> None:
        # add measurements to the yaw
        self.yaw.add_measurement(measurement)

    def add_power(self, measurement: float | NDArray[np.floating]) -> None:
        # add measurements to the power
        self.power.add_measurement(measurement)

    def get_ws(self, scaled: bool = False) -> NDArray[np.float32]:
        # get the ws measurements
        if scaled:
            # Scale the measurements
            return self._scale_val(self.ws.get_measurements(), self.ws_min, self.ws_max)
        else:
            return self.ws.get_measurements()

    def get_wd(self, scaled: bool = False) -> NDArray[np.float32]:
        # get the wd measurements
        if scaled:
            return self._scale_val(self.wd.get_measurements(), self.wd_min, self.wd_max)
        else:
            return self.wd.get_measurements()

    def get_yaw(self, scaled: bool = False) -> NDArray[np.float32]:
        # get the yaw measurements
        if scaled:
            # Scale the measurements
            return self._scale_val(
                self.yaw.get_measurements(), self.yaw_min, self.yaw_max
            )
        else:
            return self.yaw.get_measurements()

    def get_power(self, scaled: bool = False) -> NDArray[np.float32]:
        # get the power measurements
        if scaled:
            # Scale the measurements
            return self._scale_val(
                self.power.get_measurements(), 0, self.power_max
            )  # Min power is 0
        else:
            return self.power.get_measurements()

    def _scale_val(
        self, val: NDArray[np.float32], min_val: float, max_val: float
    ) -> NDArray[np.float32]:
        # Scale the value from -1 to 1
        return 2 * (val - min_val) / (max_val - min_val) - 1

    def get_measurements(self, scaled: bool = False) -> NDArray[np.float32]:
        measurements = []

        if hasattr(self, "probes"):
            # Read values from each probe, regardless of WS or TI
            probe_values = np.array(
                [
                    float(p.read()) if np.isscalar(p.read()) else float(p.read()[0])
                    for p in self.probes
                ]
            )

            if scaled:
                scaled_values = self._scale_val(
                    probe_values, self.probe_min, self.probe_max
                )
                measurements.append(scaled_values)
            else:
                measurements.append(probe_values)

        if scaled:
            measurements.extend(
                [
                    self._scale_val(self.get_ws(), self.ws_min, self.ws_max),
                    self._scale_val(self.get_wd(), self.wd_min, self.wd_max),
                    self._scale_val(self.get_yaw(), self.yaw_min, self.yaw_max),
                    self._scale_val(self.get_TI(), self.TI_min, self.TI_max),
                    self._scale_val(self.get_power(), 0, self.power_max),
                ]
            )
        else:
            measurements.extend(
                [
                    self.get_ws(),
                    self.get_wd(),
                    self.get_yaw(),
                    self.get_TI(),
                    self.get_power(),
                ]
            )

        return np.concatenate(measurements)


class FarmMes:
    """
    Class for the measurements of the farm.
    The farm stores measurements from each turbine for wind speed, wind direction, yaw angle, power
    """

    def __init__(
        self,
        n_turbines: int,
        n_probes_per_turb: dict = {},
        turb_ws: bool = True,
        turb_wd: bool = True,
        turb_TI: bool = True,
        turb_power: bool = True,
        farm_ws: bool = True,
        farm_wd: bool = True,
        farm_TI: bool = False,
        farm_power: bool = False,
        ws_current: bool = False,
        ws_rolling_mean: bool = True,
        ws_history_N: int = 1,
        ws_history_length: int = 10,
        ws_window_length: int = 10,
        wd_current: bool = False,
        wd_rolling_mean: bool = True,
        wd_history_N: int = 1,
        wd_history_length: int = 10,
        wd_window_length: int = 10,
        yaw_current: bool = False,
        yaw_rolling_mean: bool = True,
        yaw_history_N: int = 2,
        yaw_history_length: int = 30,
        yaw_window_length: int = 1,
        power_current: bool = False,
        power_rolling_mean: bool = True,
        power_history_N: int = 1,
        power_history_length: int = 10,
        power_window_length: int = 10,
        ws_min: float = 7.0,
        ws_max: float = 20.0,
        wd_min: float = 270.0,
        wd_max: float = 360.0,
        yaw_min: float = -45,
        yaw_max: float = 45,
        TI_min: float = 0.00,
        TI_max: float = 0.50,
        power_max: float = 2000000,  # 2 MW
        ti_sample_count: int = 30,
    ) -> None:
        self.n_turbines: int = n_turbines
        self.n_probes_per_turb: dict = n_probes_per_turb
        self.turb_mes: list[TurbMes] = []
        # Do we want measurements from the turbines individually?
        self.turb_ws: bool = turb_ws
        self.turb_wd: bool = turb_wd
        self.turb_TI: bool = turb_TI
        self.turb_power: bool = turb_power

        # do we want measurements from the farm, i.e. the average of the turbines
        self.farm_ws: bool = farm_ws
        # do we want measurements from the farm, i.e. the average of the turbines
        self.farm_wd: bool = farm_wd
        self.farm_TI: bool = farm_TI
        self.farm_power: bool = farm_power

        self.ws_max: float = ws_max
        self.ws_min: float = ws_min
        self.wd_min: float = wd_min
        self.wd_max: float = wd_max
        self.yaw_min: float = yaw_min
        self.yaw_max: float = yaw_max
        self.power_max: float = power_max

        if turb_TI or farm_TI:
            # If we return the TI, check the number of data points
            if ti_sample_count < 10:  # Small sample = noisy TI signal
                print(
                    f"Warning: You are only using the last {ti_sample_count} "
                    "high-frequency samples for TI calculations. "
                    "A low number might result in a noisy estimate."
                )

        # Farm observations take the same form as turbine observations
        self.farm_observed_variables = (
            farm_ws * (ws_current + ws_rolling_mean * ws_history_N)
            + farm_wd * (wd_current + wd_rolling_mean * wd_history_N)
            + farm_TI
            + farm_power * (power_current + power_rolling_mean * power_history_N)
        )

        # For each turbine, create a class of turbine measurements
        for _ in range(n_turbines):
            self.turb_mes.append(
                TurbMes(
                    ws_current=ws_current and turb_ws,
                    ws_rolling_mean=ws_rolling_mean and turb_ws,
                    ws_history_N=ws_history_N,
                    ws_history_length=ws_history_length,
                    ws_window_length=ws_window_length,
                    wd_current=wd_current and turb_wd,
                    wd_rolling_mean=wd_rolling_mean and turb_wd,
                    wd_history_N=wd_history_N,
                    wd_history_length=wd_history_length,
                    wd_window_length=wd_window_length,
                    yaw_current=yaw_current,
                    yaw_rolling_mean=yaw_rolling_mean,
                    yaw_history_N=yaw_history_N,
                    yaw_history_length=yaw_history_length,
                    yaw_window_length=yaw_window_length,
                    power_current=power_current and turb_power,
                    power_rolling_mean=power_rolling_mean and turb_power,
                    power_history_N=power_history_N,
                    power_history_length=power_history_length,
                    power_window_length=power_window_length,
                    ws_min=self.ws_min,
                    ws_max=self.ws_max,
                    wd_min=wd_min,
                    wd_max=wd_max,
                    yaw_min=yaw_min,
                    yaw_max=yaw_max,
                    TI_min=TI_min,
                    TI_max=TI_max,
                    include_TI=turb_TI,
                    power_max=power_max,
                    ti_sample_count=ti_sample_count,
                )
            )

        # Create farm measurements (used for info dict even if no farm obs)
        self.farm_mes: TurbMes = TurbMes(
            ws_current=ws_current and farm_ws,
            ws_rolling_mean=ws_rolling_mean and farm_ws,
            ws_history_N=ws_history_N,
            ws_history_length=ws_history_length,
            ws_window_length=ws_window_length,
            wd_current=wd_current and farm_wd,
            wd_rolling_mean=wd_rolling_mean and farm_wd,
            wd_history_N=wd_history_N,
            wd_history_length=wd_history_length,
            wd_window_length=wd_window_length,
            yaw_current=False,  # avoid farm yaw measurement
            yaw_rolling_mean=False,
            yaw_history_N=yaw_history_N,
            yaw_history_length=yaw_history_length,
            yaw_window_length=yaw_window_length,
            power_current=power_current and farm_power,
            power_rolling_mean=power_rolling_mean and farm_power,
            power_history_N=power_history_N,
            power_history_length=power_history_length,
            power_window_length=power_window_length,
            ws_min=self.ws_min,
            ws_max=self.ws_max,
            wd_min=wd_min,
            wd_max=wd_max,
            yaw_min=yaw_min,
            yaw_max=yaw_max,
            TI_min=TI_min,
            TI_max=TI_max,
            include_TI=farm_TI,
            power_max=power_max * n_turbines,
            n_probes_per_turb=n_probes_per_turb,
            ti_sample_count=ti_sample_count,
        )  # The max power is the sum of all the turbines

        if self.farm_TI:
            # Set get_TI function to get_TI_farm for farm TI calculations
            self.get_TI: Callable[[bool], float] = self.get_TI_farm
        else:
            # Return empty array (skipped in np.concatenate)
            self.get_TI: Callable[[bool], NDArray[np.float32]] = self.empty_np

    def empty_np(self, scaled: bool = False) -> NDArray[np.float32]:
        # Return an empty array
        return np.array([])

    def add_ws(self, measurement: NDArray[np.floating]) -> None:
        # add measurements to the ws
        for turb in self.turb_mes:
            turb.add_ws(measurement)
        if self.farm_ws:
            self.farm_mes.add_ws(np.mean(measurement))

    def add_wd(self, measurement: NDArray[np.floating]) -> None:
        # add measurements to the wd
        for turb in self.turb_mes:
            turb.add_wd(measurement)
        if self.farm_wd:
            self.farm_mes.add_wd(np.mean(measurement))

    def add_yaw(self, measurement: NDArray[np.floating]) -> None:
        # add measurements to the yaw
        for turb in self.turb_mes:
            turb.add_yaw(measurement)

    def add_power(self, measurement: NDArray[np.floating]) -> None:
        # add measurements to the power
        for turb in self.turb_mes:
            turb.add_power(measurement)
        if self.farm_power:
            self.farm_mes.add_power(
                np.sum(measurement)
            )  # Instead of mean, we sum the power

    def add_measurements(
        self,
        ws: NDArray[np.floating],
        wd: NDArray[np.floating],
        yaws: NDArray[np.floating],
        powers: NDArray[np.floating],
    ) -> None:
        # add measurements to the ws, wd and yaw in one go
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

    def max_hist(self) -> int:
        """
        Return the maximum history length of the measurements
        """

        return self.turb_mes[0].max_hist()

    def observed_variables(self) -> int:
        """
        Returns the number of observed variables
        """

        return (
            self.turb_mes[0].observed_variables() * self.n_turbines
            + self.farm_observed_variables
        )

    def get_ws_turb(self, scaled: bool = False) -> NDArray[np.float32]:
        # get the ws measurements
        return np.array(
            [turb.get_ws(scaled=scaled) for turb in self.turb_mes]
        ).flatten()

    def get_ws_farm(self, scaled: bool = False) -> NDArray[np.float32]:
        # get the ws measurements
        if scaled:
            # Scale the measurements
            return utils.scale_val(self.farm_mes.get_ws(), self.ws_min, self.ws_max)
        else:
            return self.farm_mes.get_ws()

    def get_power_turb(self, scaled: bool = False) -> NDArray[np.float32]:
        # get the power measurements
        return np.array(
            [turb.get_power(scaled=scaled) for turb in self.turb_mes]
        ).flatten()

    def get_power_farm(self, scaled: bool = False) -> NDArray[np.float32]:
        # get the power measurements
        if scaled:
            # Scale the measurements
            return utils.scale_val(
                self.farm_mes.get_power(), 0, self.power_max * self.n_turbines
            )  # Min power is 0, max is the sum of all the turbines
        else:
            return self.farm_mes.get_power()

    def get_wd_turb(self, scaled: bool = False) -> NDArray[np.float32]:
        # get the wd measurements
        return np.array(
            [turb.get_wd(scaled=scaled) for turb in self.turb_mes]
        ).flatten()

    def get_wd_farm(self, scaled: bool = False) -> NDArray[np.float32]:
        # get the wd measurements
        if scaled:
            return utils.scale_val(self.farm_mes.get_wd(), self.wd_min, self.wd_max)
        else:
            return self.farm_mes.get_wd()

    def get_TI_turb(self, scaled: bool = False) -> NDArray[np.float32]:
        # get the TI measurements
        return np.array(
            [turb.calc_TI(scaled=scaled) for turb in self.turb_mes]
        ).flatten()

    def get_TI_farm(self, scaled: bool = False) -> float:
        # Return the average value of the TI measurements
        TI_farm = self.get_TI_turb(scaled=scaled)
        return TI_farm.mean()

    def get_yaw_turb(self, scaled: bool = False) -> NDArray[np.float32]:
        # get the yaw measurements
        return np.array([turb.get_yaw(scaled) for turb in self.turb_mes]).flatten()

    def get_measurements(self, scaled: bool = False) -> NDArray[np.float32]:
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
