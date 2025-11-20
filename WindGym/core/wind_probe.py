import numpy as np


class WindProbe:
    def __init__(
        self,
        fs,
        position,
        yaw_angle,
        turbine_position,
        include_wakes=True,
        exclude_wake_from=[],
        time=None,
        probe_type="WS",
    ):
        """
        Initialize a wind speed or TI probe.

        Args:
            fs: The wind farm environment (should have `get_windspeed()` and `get_turbulence_intensity()`).
            position: (x, y, z) tuple for probe location.
            include_wakes: Whether to include wake effects in the wind calculation.
            exclude_wake_from: Turbine indices to exclude wakes from.
            time: Specific time (optional).
            probe_type: 'WS' for wind speed, 'TI' for turbulence intensity.
        """
        self.fs = fs
        self.position = position
        self.include_wakes = include_wakes
        self.exclude_wake_from = exclude_wake_from
        self.time = time
        self.probe_type = probe_type.upper()
        self.yaw_angle = yaw_angle
        self.turbine_position = turbine_position

    def read(self):
        """Read either wind speed (u, v, w) or turbulence intensity depending on probe_type."""
        if self.probe_type == "WS":
            return self.get_projected_wind_speed_toward_turbine()
        elif self.probe_type == "TI":
            return self._read_TI()
        else:
            raise ValueError(
                f"Unsupported probe_type: {self.probe_type}. Use 'WS' or 'TI'."
            )

    def _read_wind_speed(self):
        uvw = self.fs.get_windspeed(
            xyz=self.position,
            include_wakes=self.include_wakes,
            exclude_wake_from=self.exclude_wake_from,
            time=self.time,
            xarray=False,
        )
        return np.array(uvw).flatten()

    def _read_TI(self):
        return self.fs.get_turbulence_intensity(
            xyz=self.position,
            include_wake_turbulence=self.include_wakes,
        )

    def read_speed_magnitude(self):
        """Return scalar wind speed magnitude."""
        uvw = self._read_wind_speed()
        return np.linalg.norm(uvw)

    def update_position(self, new_position):
        """Move probe to a new (x, y, z) position."""
        self.position = new_position

    def get_projected_wind_speed_toward_turbine(self):
        """
        Projects the wind speed vector onto the direction from the probe to the turbine.

        Args:
            turbine_position: (x, y, z) of the turbine.

        Returns:
            Scalar wind speed component in direction from probe to turbine.
        """
        uvw = self._read_wind_speed()
        u, v = uvw[0], uvw[1]

        # Unit vector pointing from probe to turbine
        dx = self.turbine_position[0] - self.position[0]
        dy = self.turbine_position[1] - self.position[1]
        norm = np.hypot(dx, dy)

        if norm == 0:
            raise ValueError("Probe and turbine are at the same location.")

        ux = dx / norm
        uy = dy / norm

        # Dot product to project wind vector onto direction vector
        projected_speed = u * ux + v * uy
        return projected_speed

    def get_inflow_angle_to_turbine(self, degrees=False):
        """
        Returns the angle from the probe to the turbine (horizontal XY-plane),
        counter-clockwise from the x-axis.

        Args:
            turbine_position: (x, y, z) of the turbine.
            degrees: If True, return angle in degrees.

        Returns:
            Angle in radians (or degrees if requested).
        """
        dx = self.turbine_position[0] - self.position[0]
        dy = self.turbine_position[1] - self.position[1]
        angle_rad = np.arctan2(dy, dx)
        if degrees:
            return np.rad2deg(angle_rad)
        return angle_rad
