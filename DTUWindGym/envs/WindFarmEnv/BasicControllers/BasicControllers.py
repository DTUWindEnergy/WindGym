"""
This file contains the basic controllers for the wind farm environment.
These can be used either as a base controller for comparing the performance directly in hte reward function, or it can used to evaluate the farm using a "normal" base controller.

"""

import numpy as np


def local_yaw_controller(fs, yaw_step=1):
    """
    fs: Flow simulation object
    new_yaw: np.array of new yaw angles for each turbine

    This is the logic for the base controller. It just wants to move back to have zero yaw angles.
    It works on the "local" wind conditions, and tries to move the yaw angles back to zero.
    Note that it doesnt filter the local winddirections in any way, so it just moves perfectly towards the winddirection at every step.
    """
    # Fist we get the current yaw offset, in relation to the "global wind"
    yaw_baseline = fs.windTurbines.yaw

    # Then by taking the inverse tan of the wind speed components, we get the LOCAL wind direction
    wind_dir_baseline = np.rad2deg(np.arctan(fs.windTurbines.rotor_avg_windspeed(
        include_wakes=True)[1] / fs.windTurbines.rotor_avg_windspeed(include_wakes=True)[0]))

    # The desired yaw offset is the difference between the baseline yaw and the baseline wind direction
    yaw_offset = wind_dir_baseline - yaw_baseline

    # We find the direction of the yaw_action, by taking the sign
    step_dir = np.sign(yaw_offset)

    # Then we find the size of the steps. This is the minimum of the yaw_offset and the yaw_step. This is to make sure that we dont overshoot the target, and to kap the max step taken to be the yaw_step
    # this is how large the steps would be, if it was unlimited
    step_scale = np.abs(yaw_offset)
    # here we replace all values that are larger then the max, with the max
    step_scale[step_scale > yaw_step] = yaw_step

    yaw_action = step_dir * step_scale

    new_yaw = yaw_baseline + yaw_action

    return new_yaw


def global_yaw_controller(fs, yaw_step=1):
    """
    fs: Flow simulation object
    new_yaw: np.array of new yaw angles for each turbine

    This is the logic for the base controller. But now it only sees the "global" wind direction. 
    """

    # The current yaw offset, in relation to the "global wind"
    yaw_offset = fs.windTurbines.yaw

    # Frst we find the direction of the yaw_action, by taking the sign
    step_dir = np.sign(yaw_offset)

    # Then we find the size of the steps. This is the minimum of the yaw_offset and the yaw_step. This is to make sure that we dont overshoot the target, and to kap the max step taken to be the yaw_step
    # this is how large the steps would be, if it was unlimited
    step_scale = np.abs(yaw_offset)
    # here we replace all values that are larger then the max, with the max
    step_scale[step_scale > yaw_step] = yaw_step

    yaw_action = step_dir * step_scale

    new_yaw = yaw_offset - yaw_action

    return new_yaw
