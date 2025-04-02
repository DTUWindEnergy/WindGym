import unittest
import numpy as np
from WindGym.MesClass import Mes, turb_mes, farm_mes


# These dummy noise functions will be used for farm_mes.
def dummy_random_normal(self, mean, std, n):
    # Return zeros for deterministic behavior.
    return np.zeros(n, dtype=np.float32)


def dummy_return_zeros(self, mean, std, n):
    # Return zeros for deterministic behavior.
    return np.zeros(n, dtype=np.float32)


# ------------------------------------------------------------------------------
# Tests for the base measurement class "Mes"
# ------------------------------------------------------------------------------


class TestMes(unittest.TestCase):
    def test_empty_measurements(self):
        m = Mes(current=True, rolling_mean=False)
        # With no measurements added, should return an empty array.
        result = m.get_measurements()
        self.assertEqual(result.size, 0)

    def test_current_only(self):
        m = Mes(current=True, rolling_mean=False)
        # Add a measurement. In get_measurements, current is implemented by taking np.mean.
        m.add_measurement(5)
        result = m.get_measurements()
        np.testing.assert_array_almost_equal(result, np.array([5], dtype=np.float32))

    def test_rolling_mean_only(self):
        # Create a Mes that does not include the current value but only rolling means.
        # Use history_N=3 and window_length=2.
        m = Mes(
            current=False,
            rolling_mean=True,
            history_N=3,
            window_length=2,
            history_length=10,
        )
        # Add two measurements.
        m.add_measurement(2)
        m.add_measurement(4)
        result = m.get_measurements()
        # With two values [2, 4] the window (whether latest, middle or oldest) always has mean 3.
        expected = np.array([3.0, 3.0, 3.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_current_and_rolling(self):
        # Create a Mes that includes both current and rolling means.
        m = Mes(
            current=True,
            rolling_mean=True,
            history_N=3,
            window_length=1,
            history_length=10,
        )
        for i in range(1, 4):
            m.add_measurement(i)
        result = m.get_measurements()
        # current: 3; rolling: three windows each having one value 3, 2, 1.
        expected = np.array([3, 3, 2, 1], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_rolling_less(self):
        # Test the rolling functionality with a history length less than the number of measurements.
        m = Mes(
            current=False,
            rolling_mean=True,
            history_N=3,
            window_length=1,
            history_length=10,
        )
        for i in range(1, 3):
            m.add_measurement(i)
        result = m.get_measurements()
        # As we only have 1 and 2 in the history, the 2 will be repeated twice
        expected = np.array([2, 2, 1], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_call(self):
        # Test the __call__ method of the Mes class.
        m = Mes(current=True, rolling_mean=False)
        m(5)
        result = m.get_measurements()
        expected = np.array([5], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_append(self):
        # Test the append method of the Mes class.
        m = Mes(current=True, rolling_mean=False)
        m.append(5)
        result = m.get_measurements()
        expected = np.array([5], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)


# ------------------------------------------------------------------------------
# Tests for the turbine measurement class "turb_mes"
# ------------------------------------------------------------------------------


class TestTurbMes(unittest.TestCase):
    def setUp(self):
        # Create an instance with specific parameters.
        self.t = turb_mes(
            ws_current=False,
            ws_rolling_mean=True,
            ws_history_N=1,
            ws_history_length=10,
            ws_window_length=5,
            wd_current=False,
            wd_rolling_mean=True,
            wd_history_N=1,
            wd_history_length=10,
            wd_window_length=5,
            yaw_current=False,
            yaw_rolling_mean=True,
            yaw_history_N=2,
            yaw_history_length=30,
            yaw_window_length=1,
            power_current=False,
            power_rolling_mean=True,
            power_history_N=1,
            power_history_length=10,
            power_window_length=5,
            ws_min=7.0,
            ws_max=20.0,
            wd_min=270.0,
            wd_max=360.0,
            yaw_min=-45,
            yaw_max=45,
            TI_min=0.0,
            TI_max=0.5,
            include_TI=True,
            power_max=2000000,
        )

    def test_observed_variables(self):
        # For ws: 0 + (1 * 1) = 1; wd: 1; yaw: 0 + (1*2)=2; TI:1; power: 1.
        # Total expected = 1+1+2+1+1 = 6.
        self.assertEqual(self.t.observed_variables(), 6)

    def test_max_hist(self):
        # max of ws_history_length (10), wd_history_length (10), yaw_history_length (30) is 30.
        self.assertEqual(self.t.max_hist(), 30)

    def test_add_and_get_measurements(self):
        # Add one measurement to each channel.
        self.t.add_ws(10)
        self.t.add_wd(300)
        self.t.add_yaw(0)
        self.t.add_power(1000)

        ws = self.t.get_ws(scaled=False)
        wd = self.t.get_wd(scaled=False)
        yaw = self.t.get_yaw(scaled=False)
        power = self.t.get_power(scaled=False)
        TI = self.t.calc_TI(scaled=False)

        # With rolling_mean only and one measurement, the mean equals the measurement.
        np.testing.assert_array_almost_equal(ws, np.array([10], dtype=np.float32))
        np.testing.assert_array_almost_equal(wd, np.array([300], dtype=np.float32))
        np.testing.assert_array_almost_equal(yaw, np.array([0, 0], dtype=np.float32))
        np.testing.assert_array_almost_equal(power, np.array([1000], dtype=np.float32))
        # With a constant wind speed, TI (std/mean) should be 0.
        np.testing.assert_array_almost_equal(TI, np.array([0], dtype=np.float32))

    def test_scaled_values(self):
        # Test the _scale_val method.
        scaled_min = self.t._scale_val(np.array([7], dtype=np.float32), 7, 20)
        scaled_max = self.t._scale_val(np.array([20], dtype=np.float32), 7, 20)
        mid_val = self.t._scale_val(
            np.array([13.5], dtype=np.float32), 7, 20
        )  # expected 0
        np.testing.assert_array_almost_equal(
            scaled_min, np.array([-1], dtype=np.float32)
        )
        np.testing.assert_array_almost_equal(
            scaled_max, np.array([1], dtype=np.float32)
        )
        np.testing.assert_array_almost_equal(mid_val, np.array([0], dtype=np.float32))

    def test_get_measurements(self):
        # Add one measurement to each channel and test concatenation.
        self.t.add_ws(10)
        self.t.add_wd(300)
        self.t.add_yaw(0)
        self.t.add_power(1000)
        meas = self.t.get_measurements(scaled=False)
        # The expected vector is concatenation of ws (1), wd (1), yaw (2), TI (1), power (1) → length 5.
        self.assertEqual(meas.shape[0], 6)


# ------------------------------------------------------------------------------
# Tests for the farm measurement class "farm_mes"
# ------------------------------------------------------------------------------
# Since farm_mes inherits from WindEnv (which is not provided), we patch its noise functions.
# The __init__ of farm_mes uses a noise parameter to choose between _random_normal and _return_zeros.
# We override these methods for testing to yield deterministic (zero) noise.


class TestFarmMes(unittest.TestCase):
    def setUp(self):
        # Use 2 turbines for testing.
        self.n_turbines = 2
        self.f = farm_mes(
            n_turbines=self.n_turbines,
            noise="None",
            turb_ws=True,
            turb_wd=True,
            turb_TI=True,
            turb_power=True,
            farm_ws=True,
            farm_wd=True,
            farm_TI=True,
            farm_power=True,
            ws_current=True,
            ws_rolling_mean=True,
            ws_history_N=1,
            ws_history_length=10,
            ws_window_length=5,
            wd_current=True,
            wd_rolling_mean=True,
            wd_history_N=1,
            wd_history_length=10,
            wd_window_length=5,
            yaw_current=True,
            yaw_rolling_mean=True,
            yaw_history_N=1,
            yaw_history_length=10,
            yaw_window_length=5,
            power_current=True,
            power_rolling_mean=True,
            power_history_N=1,
            power_history_length=10,
            power_window_length=5,
        )
        # Patch the noise functions to return zeros.
        self.f._return_zeros = dummy_return_zeros.__get__(self.f, farm_mes)
        self.f._random_normal = dummy_random_normal.__get__(self.f, farm_mes)
        self.f._add_noise = self.f._return_zeros

    def test_observed_variables(self):
        # For each turbine:
        #   ws: current True + rolling_mean True*1 = 2
        #   wd: similarly 2,
        #   yaw: current True + rolling_mean True*1 = 2,
        #   TI: 1 (if included),
        #   power: current True + rolling_mean True*1 = 2.
        # So per turbine: 2+2+2+1+2 = 9.
        # Farm-level measurements (from the farm_mes attribute):
        #   farm_ws: 2, farm_wd: 2, farm_TI: 1, farm_power: 2 → total = 7.
        # Total expected = 9 * n_turbines + 7 = 9*2 + 7 = 25.
        self.assertEqual(self.f.observed_variables(), 25)

    def test_max_hist(self):
        # max_hist is taken from the first turbine's measurements:
        # max of ws_history_length (10), wd_history_length (10), yaw_history_length (10) → 10.
        self.assertEqual(self.f.max_hist(), 10)

    def test_add_ws(self):
        speeds = np.array([10.0, 10.0], dtype=np.float32)
        self.f.add_ws(speeds)
        speeds = np.array([12.0, 12.0], dtype=np.float32)
        self.f.add_ws(speeds)
        # For each turbine, check that ws measurement was added.
        for turb in self.f.turb_mes:
            m = turb.ws.get_measurements()
            np.testing.assert_array_almost_equal(
                m, np.array([12, 11], dtype=np.float32)
            )
        # Also check farm ws measurement.
        farm_ws = self.f.farm_mes.ws.get_measurements()
        np.testing.assert_array_almost_equal(
            farm_ws, np.array([12, 11], dtype=np.float32)
        )

    def test_add_wd(self):
        wd = np.array([300.0, 300.0], dtype=np.float32)
        self.f.add_wd(wd)
        wd = np.array([310.0, 310.0], dtype=np.float32)
        self.f.add_wd(wd)
        for turb in self.f.turb_mes:
            m = turb.wd.get_measurements()
            np.testing.assert_array_almost_equal(
                m, np.array([310, 305], dtype=np.float32)
            )
        farm_wd = self.f.farm_mes.wd.get_measurements()
        np.testing.assert_array_almost_equal(
            farm_wd, np.array([310, 305], dtype=np.float32)
        )

    def test_add_yaw(self):
        yaw = np.array([-5.0, -5.0], dtype=np.float32)
        self.f.add_yaw(yaw)
        yaw = np.array([5.0, 5.0], dtype=np.float32)
        self.f.add_yaw(yaw)
        for turb in self.f.turb_mes:
            m = turb.yaw.get_measurements()
            np.testing.assert_array_almost_equal(
                m, np.array([5.0, 0], dtype=np.float32)
            )

    def test_add_power(self):
        power = np.array([1000.0, 1000.0], dtype=np.float32)
        self.f.add_power(power)
        power = np.array([1500.0, 1500.0], dtype=np.float32)
        self.f.add_power(power)
        for turb in self.f.turb_mes:
            m = turb.power.get_measurements()
            np.testing.assert_array_almost_equal(
                m, np.array([1500, 1250], dtype=np.float32)
            )
        farm_power = self.f.farm_mes.power.get_measurements()
        # For the farm, power is summed over turbines.
        np.testing.assert_array_almost_equal(
            farm_power, np.array([3000, 2500], dtype=np.float32)
        )

    def test_add_measurements(self):
        ws = np.array([10.0, 10.0], dtype=np.float32)
        wd = np.array([300.0, 300.0], dtype=np.float32)
        yaws = np.array([0.0, 0.0], dtype=np.float32)
        powers = np.array([1000.0, 1000.0], dtype=np.float32)
        self.f.add_measurements(ws, wd, yaws, powers)

        ws = np.array([12.0, 12.0], dtype=np.float32)
        wd = np.array([310.0, 310.0], dtype=np.float32)
        yaws = np.array([0.0, 0.0], dtype=np.float32)
        powers = np.array([1500.0, 1500.0], dtype=np.float32)
        self.f.add_measurements(ws, wd, yaws, powers)
        # Check one of the turbines’ ws measurement.
        for turb in self.f.turb_mes:
            m_ws = turb.ws.get_measurements()
            np.testing.assert_array_almost_equal(
                m_ws, np.array([12, 11], dtype=np.float32)
            )
        # Check farm-level ws measurement.
        m_farm_ws = self.f.farm_mes.ws.get_measurements()
        np.testing.assert_array_almost_equal(
            m_farm_ws, np.array([12, 11], dtype=np.float32)
        )

    # def test_getters_scaled(self):
    #     # Add some measurements.
    #     ws = np.array([10.0, 12.0], dtype=np.float32)
    #     wd = np.array([300.0, 310.0], dtype=np.float32)
    #     yaws = np.array([5.0, -5.0], dtype=np.float32)
    #     powers = np.array([1000.0, 1500.0], dtype=np.float32)
    #     self.f.add_measurements(ws, wd, yaws, powers)
    #     # Test the getter methods with scaling enabled.
    #     ws_turb_scaled = self.f.get_ws_turb(scaled=True)
    #     ws_farm_scaled = self.f.get_ws_farm(scaled=True)
    #     wd_turb_scaled = self.f.get_wd_turb(scaled=True)
    #     wd_farm_scaled = self.f.get_wd_farm(scaled=True)
    #     yaw_turb_scaled = self.f.get_yaw_turb(scaled=True)
    #     power_turb_scaled = self.f.get_power_turb(scaled=True)
    #     power_farm_scaled = self.f.get_power_farm(scaled=True)
    #     # As a simple check, verify that scaling maps the lower bound to -1.
    #     scaled_val = self.f._scale_val(np.array([7], dtype=np.float32), 7, 20)
    #     np.testing.assert_array_almost_equal(scaled_val, np.array([-1], dtype=np.float32))

    def test_get_TI(self):
        # For constant wind speed measurements, TI should be zero.
        # Clear previous measurements first.
        self.f.farm_mes.ws.measurements.clear()
        for turb in self.f.turb_mes:
            turb.ws.measurements.clear()
            turb.ws.add_measurement(10)
        ti_turb = self.f.get_TI_turb(scaled=False)
        np.testing.assert_array_almost_equal(
            ti_turb, np.array([0, 0], dtype=np.float32)
        )
        ti_farm = self.f.get_TI_farm(scaled=False)
        self.assertAlmostEqual(ti_farm, 0)

    def test_get_measurements(self):
        # Add measurements to both turbine and farm measurements.
        ws = np.array([10.0, 12.0], dtype=np.float32)
        wd = np.array([300.0, 310.0], dtype=np.float32)
        yaws = np.array([0.0, 0.0], dtype=np.float32)
        powers = np.array([1000.0, 1500.0], dtype=np.float32)
        self.f.add_measurements(ws, wd, yaws, powers)
        meas = self.f.get_measurements(scaled=False)
        # Check that the final measurement vector is one-dimensional and nonempty.
        self.assertTrue(meas.ndim == 1 and meas.size > 0)


if __name__ == "__main__":
    unittest.main()
