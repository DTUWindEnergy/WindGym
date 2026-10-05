"""HAWC2 load channels for ``eval_single_fast(return_loads=True)``.

``gtsdf`` is imported from ``wetb.gtsdf`` here, the same module object
``WindGym.agent_eval`` re-exports, so ``patch("WindGym.agent_eval.gtsdf.load")``
keeps working.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import xarray as xr
from wetb.gtsdf import gtsdf

# Dataset variable -> HAWC2 result-file column. Insertion order is the
# variable order of the returned dataset.
HAWC2_LOAD_COLUMNS = OrderedDict(
    [
        ("Blade_Mx", 19),
        ("Blade_My", 20),
        ("Tower_Mx", 28),
        ("Tower_My", 29),
        ("Ae_rot_torque", 10),
        ("Ae_rot_power", 11),
        ("Ae_rot_thrust", 12),
        ("WSP_gl_coo_Vx", 13),
        ("WSP_gl_coo_Vy", 14),
        ("WSP_gl_coo_Vz", 15),
        ("yaw_a", 112),
    ]
)

LOAD_DIMS = ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step")


def load_hawc2_loads_dataset(env, *, ws, wd, ti, turbbox, model_step) -> xr.Dataset:
    """Read each turbine's HAWC2 result file and stack the load channels.

    The dataset is 7-D (no ``deterministic`` dim), the time vector comes from
    turbine 0 and the arrays keep the result file's dtype.
    """
    n_turb = env.n_turb
    # First make sure we have written the lates results
    env.wts.h2.write_output()  # I am not sure this is needed tho

    all_data = []
    # For each turbine read the data and put in into an array
    for i in range(n_turb):
        file_name = env.wts.htc_lst[i].output.filename.values[0] + ".hdf5"
        test_string = env.wts.htc_lst[i].modelpath + file_name
        time, data, info = gtsdf.load(test_string)

        # Store each turbine's data in a dictionary
        d = {name: data[:, col] for name, col in HAWC2_LOAD_COLUMNS.items()}
        d["time"] = time
        all_data.append(d)

    # Assuming all turbines share the same time vector
    time = all_data[0]["time"]

    # Stack data into arrays with shape (time, turbine) and reshape to the
    # dataset dimensions
    data_vars = {}
    for name in HAWC2_LOAD_COLUMNS:
        arr = np.stack([d[name] for d in all_data]).T
        data_vars[name] = (
            LOAD_DIMS,
            arr.reshape(time.shape[0], n_turb, 1, 1, 1, 1, 1),
        )

    # Create xarray dataset with 'turb' and 'time' dimensions
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "ws": np.array([ws]),
            "wd": np.array([wd]),
            "turb": np.arange(n_turb),
            "time": time,
            "TI": np.array([ti]),
            "turbbox": [turbbox],
            "model_step": np.array([model_step]),
        },
    )


def close_hawc2_and_cleanup(env, *, cleanup: bool, baseline_comp: bool) -> None:
    """Close the HAWC2 connections and (optionally) delete the case folders + flow state."""
    # To make sure that the turbulence box is removed from memory, we set the current timestep to be equal to the max, and then do one last step.
    # This clears the turbulence box from memory, and makes sure that we dont have any issues with the turbulence box being in memory.
    env.wts.h2.close()
    if baseline_comp:
        env.wts_baseline.h2.close()

    if cleanup:
        env._deleteHAWCfolder()
        env.fs = None
        env.site = None
        env.farm_measurements = None
        del env.fs
        del env.site
        del env.farm_measurements

        if baseline_comp:
            env.fs_baseline = None
            env.site_base = None
            del env.fs_baseline
            del env.site_base
