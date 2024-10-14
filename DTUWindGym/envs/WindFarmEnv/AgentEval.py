import xarray as xr
import numpy as np
# import concurrent.futures
# import multiprocessing
from pathos.pools import ProcessPool
import time
from dynamiks.views import XYView, EastNorthView
from dynamiks.visualizers.flow_visualizers import Flow2DVisualizer
from py_wake.utils.plotting import setup_plot
import os
import matplotlib.pyplot as plt
from collections import deque 
from py_wake.wind_turbines import WindTurbines as WindTurbinesPW
from matplotlib.transforms import Bbox
"""
This class is able to evaluate an agent on the PyWake environment.
The class is made to evaluate the agent for multiple wind directions, and then save a xarray dataset with the results.
TODO: Finish the class so that it can plot the results, and save the results to a file. maybe? 
TODO: We could add in a check that the agent has already been evaluated on a given condition. if yes, then we dont need to simulate it again.
TODO: Add a function to animate the results.
TODO: parallelize the evaluation in eval_multiple()
"""
class AgentEval():
    def __init__(self, env=None, model=None, name = "NoName", t_sim = 1000):
        #Initialize the evaluater with some default values.
        self.ws = 10.0
        self.ti = 0.05
        self.wd = 270 
        self.yaw = 0.0
        self.turbbox = "Default"

        self.t_sim = t_sim

        self.winddirs = [270]
        self.windspeeds = [10]
        self.turbintensities = [0.05]
        self.turbboxes = ["Default"]

        self.multiple_eval = False #Flag if multiple_eval has been called.
        self.env = env
        self.model = model
        self.name = name
        
    def set_conditions(self, winddirs:list=[], windspeeds:list=[], turbintensities:list=[], turbboxes:list=["Default"]):
        #Update the conditions for the evaluation.
        if winddirs:
            self.winddirs = winddirs
        if windspeeds:
            self.windspeeds = windspeeds
        if turbintensities:
            self.turbintensities = turbintensities
        if turbboxes:
            self.turbboxes = turbboxes

    def set_condition(self, ws=None, ti=None, wd=None, yaw=None, turbbox=None):
        #Set the conditions for the individual evaluation, and then update the env with these values. 
        if ws != None:
            self.ws = ws
        if ti != None:
            self.ti = ti
        if wd != None:
            self.wd = wd
        if yaw != None:
            self.yaw = yaw
        if turbbox != None:
            self.turbbox = turbbox

        self.set_env_vals()

    def set_env_vals(self):
        #Update the environment with the new conditions
        # First we initialize the environment with the specified conditions
        self.env.set_yaw_vals(self.yaw) #Specified yaw vals
        self.env.set_wind_vals(ws = self.ws, ti=self.ti, wd = self.wd) #Set the wind values, used for initialization
        if self.turbbox != "Default":
            #NOTE you must make sure that the self.turbbox is set to a path with a turbulence box file.
            #Also it must point to a specific file, and not a folder.
            self.env.update_tf(self.turbbox)   #Here we can specify a path for the turbulence box to be used.
        
    def update_env(self, env):
        #Update the environment with the new conditions 
        self.env = env

    def update_model(self, model):
        #Update the model with the new conditions 
        # Can be used if model=None in the inital call. 
        self.model = model

    def eval_single(self, save_figs=False, scale_obs=None, debug=False):
        #Evaluate the agent on a singe run. 
        # We should run the simulation for x time steps to allow the agent to take it's actions, and then simulate for y time steps to evaluate the "steady state" performance.
        #If save_figs is True, then we should save the figures to a file.
        if not isinstance(scale_obs, list): #if not a list, make it one
            scaling = [scale_obs]
        if debug: #If debug, do both. 
            scaling = [True, False]
        
        if self.model is None:
            AssertionError("You need to specify a model to evaluate the agent.")

        #Unpack some variables, to make the code more readable
        time = self.t_sim               #Time to simulate
        n_turb = self.env.n_turb        #Number of turbines
        n_ws = 1                        #Number of wind speeds to simulate
        n_wd = 1                        #Number of wind direction simulate
        n_turbbox = 1                   #Number of turbulence boxes to simulate
        n_TI = 1                        #Number of turbulence intensities to simulate

        #Initialize the arrays to store the results
        #Power at farm level and turbine level, for both the agent and the baseline farm
        powerF_a = np.zeros((time))
        powerF_b = np.zeros((time))
        powerT_a = np.zeros((time, n_turb))
        powerT_b = np.zeros((time, n_turb))
        #Yaw angles for the agent and the baseline farm
        yaw_a = np.zeros((time, n_turb))
        yaw_b = np.zeros((time, n_turb))
        #Wind speeds at the rotor for the agent and the baseline farm
        ws_a = np.zeros((time, n_turb))
        ws_b = np.zeros((time, n_turb))
        #Time array
        time_plot = np.zeros((time))
        #Percentage increase in power output, and reward
        pct_inc = np.zeros((time))
        rew_plot = np.zeros((time))

        #Initialize the environment
        obs, info = self.env.reset()

        #After reset, if the model has a pywakeagent attribute, we can use this to set the yaw angles.
        if hasattr(self.model, "pywakeagent"):
            self.model.update_wind(self.env.ws, self.env.wd, self.env.ti)
            self.model.predict(obs, deterministic=True)[0]

        #Put the initial values in the arrays
        powerF_a[0] = self.env.fs.windTurbines.power().sum()
        powerF_b[0] = self.env.fs_baseline.windTurbines.power().sum()
        powerT_a[0] = self.env.fs.windTurbines.power()
        powerT_b[0] = self.env.fs_baseline.windTurbines.power()

        yaw_a[0] = info["yaw angles agent"]
        yaw_b[0] = info["yaw angles base"]
        ws_a[0] = np.linalg.norm(self.env.fs.windTurbines.rotor_avg_windspeed(include_wakes=True), axis=0)
        ws_b[0] = np.linalg.norm(self.env.fs_baseline.windTurbines.rotor_avg_windspeed(include_wakes=True), axis=0)
        time_plot[0] = self.env.fs.time
        rew_plot[0] = 0.0  #There is no reward at the first time step, so we just set it to zero.
        pct_inc[0] = ((powerF_a[0]-powerF_b[0]) /powerF_b[0]) * 100 #Percentage increase in power output. This should be zero (or close to zero) at the first time step.

        #If save_figs is True, initalize some parameters here.
        if save_figs:
            FOLDER='./Temp_Figs_{}_ws{}_wd{}/'.format(self.name, self.env.ws, self.wd)
            if not os.path.exists(FOLDER):
                os.makedirs(FOLDER)
            max_deque = 70
            time_deq = deque(maxlen=max_deque)
            pow_deq = deque(maxlen=max_deque)
            yaw_deq = deque(maxlen=max_deque)
            ws_deq = deque(maxlen=max_deque)

            time_deq.append(time_plot[0])
            pow_deq.append(powerF_a[0])
            yaw_deq.append(yaw_a[0])
            ws_deq.append(ws_a[0])
            #These are used for y limits on the plot. 
            pow_max = powerF_a[0]*1.2
            pow_min = powerF_a[0]*0.8
            yaw_max = 5
            yaw_min = -5
            ws_max = self.env.ws+2
            ws_min = 3


            a = np.linspace(-200 + min(self.env.x_pos), 200 + max(self.env.x_pos), 200)
            b = np.linspace(-200 + min(self.env.y_pos), 200 + max(self.env.y_pos), 200)

        #Run the simulation
        for i in range(1,time):

            action = self.model.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, info = self.env.step(action)

            #Put the values in the arrays
            powerF_a[i] = self.env.fs.windTurbines.power().sum()
            powerF_b[i] = self.env.fs_baseline.windTurbines.power().sum()
            powerT_a[i] = self.env.fs.windTurbines.power()
            powerT_b[i] = self.env.fs_baseline.windTurbines.power()

            yaw_a[i] = info["yaw angles agent"]
            yaw_b[i] = info["yaw angles base"]

            ws_a[i] = np.linalg.norm(self.env.fs.windTurbines.rotor_avg_windspeed(include_wakes=True), axis=0)
            ws_b[i] = np.linalg.norm(self.env.fs_baseline.windTurbines.rotor_avg_windspeed(include_wakes=True), axis=0)
            time_plot[i] = self.env.fs.time
            rew_plot[i] = reward  #
            pct_inc[i] = ((powerF_a[i]-powerF_b[i]) /powerF_b[i]) * 100 #Percentage increase in power output. This should be zero (or close to zero) at the first time step.

            if save_figs:
                time_deq.append(time_plot[i])
                pow_deq.append(powerF_a[i])
                yaw_deq.append(yaw_a[i])
                ws_deq.append(ws_a[i])

                # fig = plt.figure(constrained_layout=False, figsize=(12, 7.5))
                fig = plt.figure(figsize=(12, 7.5))
                # Create a grid spec with 2 rows and 2 columns
                # gs = fig.add_gridspec(3, 2, width_ratios=[3, 1])

                ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)

                # ax1 = fig.add_subplot(gs[:, 0])
                view = XYView(z=70, x= a, y=b, ax=fig.gca(), adaptive=False)

                wt = self.env.fs.windTurbines
                x_turb, y_turb = wt.positions_xyz(self.env.fs.wind_direction, self.env.fs.center_offset)[:2]
                yaw, tilt = wt.yaw_tilt()
            
                uvw = self.env.fs.get_windspeed(view, include_wakes=True, xarray=True)

                plt.pcolormesh(uvw.x.values, uvw.y.values, uvw[0].T, shading="nearest", vmin=3, vmax=self.env.ws+2)  #[0] is the u component of the wind speed
                # test = plt.colorbar()
                plt.colorbar().set_label('Wind speed, u [m/s]')
                WindTurbinesPW.plot_xy(wt, x_turb, y_turb, types=wt.types, wd=self.env.fs.wind_direction, ax=ax1, yaw=yaw, tilt=tilt)

                ax1.set_title('Flow field at {} s'.format(self.env.fs.time))
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())

                ax2 = plt.subplot2grid((3, 3), (0, 2), )
                ax3 = plt.subplot2grid((3, 3), (1, 2), )
                ax4 = plt.subplot2grid((3, 3), (2, 2), )

                # ax2 = fig.add_subplot(gs[0, 1])
                ax2.plot(time_deq, pow_deq, color='orange')
                ax2.set_title('Farm power [W]')
                # ax2.legend()

                # ax3 = fig.add_subplot(gs[1, 1])
                ax3.plot(time_deq, yaw_deq, label=np.arange(n_turb))
                ax3.set_title('Turbine yaws [deg]')
                ax3.legend(loc='upper left')

                # ax4 = fig.add_subplot(gs[2, 1])
                ax4.plot(time_deq, ws_deq, label=np.arange(n_turb))
                ax4.set_title('Rotor windspeeds [m/s]')
                ax4.set_xlabel('Time [s]')
                # ax4.legend()

                ax2.set_xlim(time_deq[0], time_deq[-1])
                ax3.set_xlim(time_deq[0], time_deq[-1])
                ax4.set_xlim(time_deq[0], time_deq[-1])

                pow_max = max(pow_max, powerF_a[i]*1.2)
                pow_min = min(pow_min, powerF_a[i]*0.8)
                yaw_max = max(yaw_max, max(yaw_a[i])*1.2)
                yaw_min = min(yaw_min, min(yaw_a[i])*1.2) #This value can be negative, so we multiply 1.2, instead of 0.8
                ws_max = max(ws_max, max(ws_a[i])*1.2)
                ws_min = min(ws_min, min(ws_a[i])*0.8)
                             
                #Set the y limits for the plots. If we go over/under the limits, the plot will adjust the limits.
                ax2.set_ylim(pow_min, pow_max)
                ax3.set_ylim(yaw_min, yaw_max)
                ax4.set_ylim(ws_min, ws_max)
                ax2.set_xticks([])
                ax3.set_xticks([])

                #Set the number of ticks on the x-axis to 5
                ax4.locator_params(axis='x', nbins=5)

                img_name = FOLDER + 'img_{:05d}.png'.format(i)
                #Set the plot to be compact:
                # plt.tight_layout()

                #Add sensor data to the plot. 
                for scale in scaling:
                    if scale is not None:
                        # print("scale is not None")
                        turb_ws = np.round(self.env.farm_measurements.get_ws_turb(scale),2) 
                        turb_wd = np.round(self.env.farm_measurements.get_wd_turb(scale),2) 
                        turb_TI = np.round(self.env.farm_measurements.get_TI_turb(scale),2)
                        turb_yaw = np.round(self.env.farm_measurements.get_yaw_turb(scale),2)
                        farm_ws = np.round(self.env.farm_measurements.get_ws_farm(scale),2)
                        farm_wd = np.round(self.env.farm_measurements.get_wd_farm(scale),2)
                        farm_TI = np.round(self.env.farm_measurements.get_TI(scale),2)
                        if scale:
                            text_plot = f" Agent observations scaled: \n Turbine level wind speed: {turb_ws} \n Turbine level wind direction: {turb_wd} \n Turbine level yaw: {turb_yaw} \n Turbine level TI: {turb_TI} \n Farm level wind speed: {farm_ws} \n Farm level wind direction: {farm_wd} \n Farm level TI: {farm_TI} "
                            ax1.text(1.1, 1.3, text_plot, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes)
                            # plt.text(1.1, 1.3, text_plot, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes)
                        else:
                            text_plot = f" Agent observations: \n Turbine level wind speed: {turb_ws} [m/s] \n Turbine level wind direction: {turb_wd} [deg] \n Turbine level yaw: {turb_yaw} [deg] \n Turbine level TI: {turb_TI} \n Farm level wind speed: {farm_ws} [m/s] \n Farm level wind direction: {farm_wd} [deg] \n Farm level TI: {farm_TI} "
                            ax1.text(-0.1, 1.3, text_plot, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes)
                            # plt.text(-0.1, 1.3, text_plot, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes)
                #So I coudnt figure out how to add some space to the left, so I added a white text, and then use that to stretch the plot. Whatever, it works
                ax1.text(1.95, 0.5, "Hey", verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color='white')

                plt.savefig(img_name, dpi=100, bbox_extra_artists=(ax1, ax2, ax3, ax4), bbox_inches='tight')
                plt.clf()
                plt.close('all')
                    
        self.env.close()

        #Reshape the arrays
        powerF_a = powerF_a.reshape(time, n_ws, n_wd, n_TI, n_turbbox)
        powerF_b = powerF_b.reshape(time, n_ws, n_wd, n_TI, n_turbbox)
        powerT_a = powerT_a.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox)
        powerT_b = powerT_b.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox)
        yaw_a = yaw_a.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox)
        yaw_b = yaw_b.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox)
        ws_a = ws_a.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox)
        ws_b = ws_b.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox)

        # time_plot = time_plot.reshape(time, n_ws, n_wd, n_TI, n_turbbox)
        rew_plot = rew_plot.reshape(time, n_ws, n_wd, n_TI, n_turbbox)
        pct_inc = pct_inc.reshape(time, n_ws, n_wd, n_TI, n_turbbox)

        #Then create a xarray dataset with the results
        ds = xr.Dataset(
            data_vars={
                #For agent:
                "powerF_a": (("time", "ws", "wd", "TI", "turbbox"), powerF_a), #Power for the farm: [time, turbine, ws, wd, TI, turbbox]
                "powerT_a": (("time", "turb", "ws", "wd", "TI", "turbbox"), powerT_a),  #Power pr turbine [time, ws, wd, TI, turbbox]
                "yaw_a": (("time", "turb", "ws", "wd", "TI", "turbbox"), yaw_a), #yaw is array of: [time, turbine, ws, wd, TI, turbbox]
                "ws_a": (("time", "turb", "ws", "wd", "TI", "turbbox"), ws_a),   #Ws at each turbine: [time, turbine, ws, wd, TI, turbbox]
                
                #For baseline
                "powerF_b": (("time", "ws", "wd", "TI", "turbbox"), powerF_b), #Power for the farm: [time, turbine, ws, wd, TI, turbbox]
                "powerT_b": (("time", "turb", "ws", "wd", "TI", "turbbox"), powerT_b),  #Power pr turbine [time, ws, wd, TI, turbbox]
                "yaw_b": (("time", "turb", "ws", "wd", "TI", "turbbox"), yaw_b), #yaw is array of: [time, turbine, ws, wd, TI, turbbox]
                "ws_b": (("time", "turb", "ws", "wd", "TI", "turbbox"), ws_b),   #Ws at each turbine: [time, turbine, ws, wd, TI, turbbox]

                #For environment
                "reward": (("time", "ws", "wd", "TI", "turbbox"), rew_plot), #Reward 
                "pct_inc": (("time", "ws", "wd", "TI", "turbbox"), pct_inc), #Percentage increase in power output
                
            },
            coords={
                "ws": np.array([self.ws]),
                "wd": np.array([self.wd]),
                "turb": np.arange(self.env.n_turb),
                "time": time_plot,
                "TI": np.array([self.ti]),
                "turbbox": [self.turbbox],
            },
        )
        return ds

    def eval_multiple(self, save_figs=False, scale_obs=None, debug=False):
        #Evaluate the agent on multiple runs
        #Make sure to update the set_conditions() first
        if self.multiple_eval:
            print("It looks like you have already run the multiple evaluations") #. If you want to run them again, please reinitialize the object.")
            # return self.multiple_eval_ds  #we could do something like this, to try and prevent the user from running the simulations again. ? 

        print("Running for a total of ", len(self.winddirs)*len(self.windspeeds)*len(self.turbintensities)*len(self.turbboxes), " simulations.")
        self.multiple_eval = True  #Flag that we are running multiple evaluations.

        #OLD CODE
        ds_list = []
        for winddir in self.winddirs:
            for windspeed in self.windspeeds:
                for TI in self.turbintensities:
                    for box in self.turbboxes:
                        #For all these in the loop...
                        #Set the conditions
                        self.set_condition(ws=windspeed, ti=TI, wd=winddir, turbbox=box)
                        #Run the simulation
                        ds = self.eval_single(save_figs=save_figs, scale_obs=scale_obs, debug=debug)
                        #Save the results 
                        ds_list.append(ds)
        ds_total = xr.merge(ds_list)
        self.multiple_eval_ds = ds_total
        return self.multiple_eval_ds

        ### Failed tests with multiprocessing. ):
        # ds_list = []
        # args_list = [(winddir, windspeed, TI, box, save_figs, scale_obs, debug)
        #              for winddir in self.winddirs
        #              for windspeed in self.windspeeds
        #              for TI in self.turbintensities
        #              for box in self.turbboxes]
        
        # pool = ProcessPool()

        # results = pool.amap(self.run_simulation, args_list)

        # #This should do it in parallel. Hopefully.
        # while not results.ready():
        #     time.sleep(10)
        # ds_list = results.get()

        # with multiprocessing.Pool() as pool:
        #     ds_list = pool.map(self.run_simulation, args_list)

        # ds_list = []
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        # with concurrent.futures.ProcessPoolExecutor() as executor:
            # futures = []
            # for winddir in self.winddirs:
            #     for windspeed in self.windspeeds:
            #         for TI in self.turbintensities:
            #             for box in self.turbboxes:
            #                 futures.append(executor.submit(self.run_simulation, winddir, windspeed, TI, box, save_figs, scale_obs, debug))
            #                 # futures.append(executor.submit(self.run_simulation, winddir,
            # for future in concurrent.futures.as_completed(futures):
            #     ds_list.append(future.result())

            #This might be a better way to do it, but I am not sure.
            # results = [executor.submit(self.run_simulation, wd, ws, TI, box, save_figs, scale_obs, debug) for wd in self.winddirs for ws in self.windspeeds for TI in self.turbintensities for box in self.turbboxes]
            # for f in concurrent.futures.as_completed(results):
            #     ds_list.append(f.result())

        # ds_total = xr.merge(ds_list)
        # self.multiple_eval_ds = ds_total
        # return self.multiple_eval_ds           

    def run_simulation(self, winddir, windspeed, TI, box, save_figs, scale_obs, debug):
        print("Running simulation for ws = ", windspeed, " wd = ", winddir, " TI = ", TI, " TurbBox = ", box)
        #Run a singe simulation with the specified conditions.
        #Set the conditions
        self.set_condition(ws=windspeed, ti=TI, wd=winddir, turbbox=box)
        #Run the simulation
        ds = self.eval_single(save_figs=save_figs, scale_obs=scale_obs, debug=debug)
        print("Done with simulation for ws = ", windspeed, " wd = ", winddir, " TI = ", TI, " TurbBox = ", box)
        return ds

    def plot_initial(self):
        #Plot the initial conditions of the simulation, AND the turbines with their numbering. 
        #First we need to initialize the environment with the specified conditions. 
        _, __ = self.env.reset()

        #Define the x, y and z for the plot
        x_mean = self.env.fs.windTurbines.position[0].mean()
        y_mean = self.env.fs.windTurbines.position[1].mean()
        x_range = self.env.fs.windTurbines.position[0].max() - self.env.fs.windTurbines.position[0].min()
        y_range = self.env.fs.windTurbines.position[1].max() - self.env.fs.windTurbines.position[1].min()
        h = self.env.fs.windTurbines.hub_height()[0]

        ax1,ax2 = plt.subplots(1,2, figsize=(10,4))[1]

        #plot in one way
        self.env.fs.show(view=XYView(x=np.linspace(x_mean - x_range,x_mean+x_range),y=np.linspace(y_mean - y_range, y_mean + y_range),z=h, ax=ax1), 
                                        flowVisualizer = Flow2DVisualizer(color_bar=False), show=False)
        #plot in another way
        self.env.fs.show(view=EastNorthView(east=np.linspace(x_mean - x_range,x_mean+x_range),north=np.linspace(y_mean - y_range, y_mean + y_range),z=h, ax=ax2),
                                        flowVisualizer = Flow2DVisualizer(color_bar=False), show=False)
        setup_plot(ax=ax1, title=f'Rotated view, {self.env.wd} deg', xlabel='x [m]', ylabel='y [m]', grid=False)
        setup_plot(ax=ax2, title=f'Alligned view, {self.env.wd} deg', xlabel='east [m]', ylabel='north [m]', grid=False)    
                
    def plot_performance(self):
        #Plot the performance of the agent, and the baseline farm. 
        # We could plot the power output, the wind speed, the wind direction, the yaw angles, the turbulence intensity, the wake losses, etc.
        # The return is a plot of the performance metrics. 
        print("Not implemented yet")
        

    def save_performance(self):
        #Save the xarray to a file. For now just on the main path.
        if self.multiple_eval:
            self.multiple_eval_ds.to_netcdf(self.name + "_eval.nc")
        else:
            print("It doenst look like you have any data to save my guy")
        
    def load_performance(self, path):
        #Load the performance metrics from a file. 
        #I dont know why you would do this, but now you can.
        self.multiple_eval_ds = xr.open_dataset(path)
        self.multiple_eval = True


    def plot_power_farm(self, WSS, WDS, avg_n=10, TI=0.07, TURBBOX="Default", axs=None, save=False):
        data = self.multiple_eval_ds #Just for easier writing
        if axs is None:
            fig, axs = plt.subplots(len(WSS), len(WDS), figsize=(4*int(len(WDS)), 3*int(len(WSS))), sharey=True)
        else:
            fig = axs[0,0].get_figure()

        for j, WS in enumerate(WSS):
            for i, wd in enumerate(WDS):
                data.sel(ws=WS).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).powerF_a.rolling(time=avg_n, center=True).mean().dropna("time").plot.line(x='time', label='Agent', ax=axs[j,i])
                data.sel(ws=WS).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).powerF_b.rolling(time=avg_n, center=True).mean().plot.line(x='time', label="Baseline", ax=axs[j,i]) # 
                
                if j == 0: #Only set the top row to have a title
                    axs[j,i].set_title(f"WD ={wd} [deg]")     
                else:
                    axs[j,i].set_title("")
                if i == 0: #Only set the left column to have a y-label
                    axs[j,i].set_ylabel(f"WS ={WS} [m/s]")
                else:
                    axs[j,i].set_ylabel("")

                axs[j,i].set_xlabel("")
                axs[j,i].grid()
                x_start = data.sel(ws=WS).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).powerF_a.rolling(time=avg_n, center=True).mean().dropna('time').time.values.min()
                x_end = data.sel(ws=WS).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).powerF_a.rolling(time=avg_n, center=True).mean().dropna('time').time.values.max()
                axs[j,i].set_xlim(x_start, x_end)
        axs[0,1].legend()
        fig.suptitle(f"Power output for agent and baseline, WS = {WSS}, WD = {WDS}, TI = {TI}, TurbBox = {TURBBOX}", fontsize=15, fontweight='bold')
        fig.supylabel("Power [W]", fontsize=15, fontweight='bold')
        fig.supxlabel("Time [s]", fontsize=15, fontweight='bold')
        plt.tight_layout()
        if save:
            plt.savefig(self.name + "_power_farm.png")
        return fig, axs

    def plot_farm_inc(self, WSS, WDS, avg_n=10, TI=0.07, TURBBOX="Default", axs=None, save=False):
        data = self.multiple_eval_ds #Just for easier writing
        if axs is None:
            fig, axs = plt.subplots(len(WSS), len(WDS), figsize=(4*int(len(WDS)), 3*int(len(WSS))), sharey=True)
        else:
            fig = axs[0,0].get_figure()

        for j, WS in enumerate(WSS):
            for i, wd in enumerate(WDS):
                data.sel(ws=WS).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).pct_inc.rolling(time=avg_n, center=True).mean().dropna("time").plot.line(x='time', ax=axs[j,i])
                if j == 0: #Only set the top row to have a title
                    axs[j,i].set_title(f"WD ={wd} [deg]")     
                else:
                    axs[j,i].set_title("")
                if i == 0: #Only set the left column to have a y-label
                    axs[j,i].set_ylabel(f"WS ={WS} [m/s]")
                else:
                    axs[j,i].set_ylabel("")

                axs[j,i].set_xlabel("")
                axs[j,i].grid()
                x_start = data.sel(ws=WS).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).pct_inc.rolling(time=avg_n, center=True).mean().dropna('time').time.values.min()
                x_end = data.sel(ws=WS).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).pct_inc.rolling(time=avg_n, center=True).mean().dropna('time').time.values.max()
                axs[j,i].set_xlim(x_start, x_end)

        fig.suptitle(f"Power increase, WS = {WSS}, WD = {WDS}, TI = {TI}, TurbBox = {TURBBOX}", fontsize=15, fontweight='bold')
        fig.supylabel("Power increase [%]", fontsize=15, fontweight='bold')
        fig.supxlabel("Time [s]", fontsize=15, fontweight='bold')
        plt.tight_layout()
        if save:
            plt.savefig(self.name + "_power_farm.png")
        return fig, axs

    def plot_power_turb(self, ws, WDS, avg_n=10, TI=0.07, TURBBOX="Default", axs=None, save=False):
        data = self.multiple_eval_ds #Just for easier writing
        n_turb = len(data.turb.values)  #The number of turbines in the farm
        n_wds = len(WDS)  #The number of wind directions we are looking at

        if axs is None:
            fig, axs = plt.subplots(n_turb, n_wds, figsize=(18, 9), sharex=True, sharey=True)
        else:
            fig = axs[0,0].get_figure()

        for i in range(n_turb):
            for j, wd in enumerate(WDS):
                data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(turb=i).powerT_a.rolling(time=avg_n, center=True).mean().dropna("time").plot.line(x='time', label='Agent', ax=axs[i, j])
                data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(turb=i).powerT_b.rolling(time=avg_n, center=True).mean().plot.line(x='time', label="Baseline", ax=axs[i, j]) #
                axs[i, j].set_title(f"WD ={wd}, Turbine {i}")
                
                x_start = data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(turb=0).powerT_a.rolling(time=avg_n, center=True).mean().dropna('time').time.values.min()
                x_end = data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(turb=0).powerT_a.rolling(time=avg_n, center=True).mean().dropna('time').time.values.max()

                axs[i, j].grid()
                axs[i, j].set_xlim(x_start, x_end)
                axs[i, j].set_ylabel(" ")
                axs[i, j].set_xlabel(" ")
                
        fig.supylabel("Power [W]", fontsize=15, fontweight='bold')
        fig.supxlabel("Time [s]", fontsize=15, fontweight='bold')
        fig.suptitle(f"Power output pr turbine for agent and baseline, ws = {ws}, WD = {WDS}, TI = {TI}, TurbBox = {TURBBOX}", fontsize=15, fontweight='bold')
        axs[0,0].legend()
        plt.tight_layout()
        if save:
            plt.savefig(self.name + "_power_farm.png")
        return fig, axs
    
    def plot_yaw_turb(self, ws, WDS, avg_n=10, TI=0.07, TURBBOX="Default", axs=None, save=False):
        data = self.multiple_eval_ds #Just for easier writing
        n_turb = len(data.turb.values)  #The number of turbines in the farm
        n_wds = len(WDS)  #The number of wind directions we are looking at

        if axs is None:
            fig, axs = plt.subplots(n_turb, n_wds, figsize=(18, 9), sharex=True, sharey=True)
        else:
            fig = axs[0,0].get_figure()

        for i in range(n_turb):
            for j, wd in enumerate(WDS):
                data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(turb=i).yaw_a.rolling(time=avg_n, center=True).mean().dropna("time").plot.line(x='time', label='Agent', ax=axs[i, j])
                data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(turb=i).yaw_b.rolling(time=avg_n, center=True).mean().plot.line(x='time', label="Baseline", ax=axs[i, j]) #
                axs[i, j].set_title(f"WD ={wd}, Turbine {i}")
                
                x_start = data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(turb=0).yaw_a.rolling(time=avg_n, center=True).mean().dropna('time').time.values.min()
                x_end = data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(turb=0).yaw_a.rolling(time=avg_n, center=True).mean().dropna('time').time.values.max()

                axs[i, j].grid()
                axs[i, j].set_xlim(x_start, x_end)
                axs[i, j].set_ylabel(" ")
                axs[i, j].set_xlabel(" ")
                
        fig.supylabel("Yaw offset [deg]", fontsize=15, fontweight='bold')
        fig.supxlabel("Time [s]", fontsize=15, fontweight='bold')
        fig.suptitle(f"Yaw angle pr turbine for agent and baseline, ws = {ws}, WD = {WDS}, TI = {TI}, TurbBox = {TURBBOX}", fontsize=15, fontweight='bold')
        axs[0,0].legend()
        plt.tight_layout()
        if save:
            plt.savefig(self.name + "_yaw_farm.png")
        return fig, axs
    
    def plot_speed_turb(self, ws, WDS, avg_n=10, TI=0.07, TURBBOX="Default", axs=None, save=False):
        data = self.multiple_eval_ds #Just for easier writing
        n_turb = len(data.turb.values)  #The number of turbines in the farm
        n_wds = len(WDS)  #The number of wind directions we are looking at

        if axs is None:
            fig, axs = plt.subplots(n_turb, n_wds, figsize=(18, 9), sharex=True, sharey=True)
        else:
            fig = axs[0,0].get_figure()

        for i in range(n_turb):
            for j, wd in enumerate(WDS):
                data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(turb=i).ws_a.rolling(time=avg_n, center=True).mean().dropna("time").plot.line(x='time', label='Agent', ax=axs[i, j])
                data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(turb=i).ws_b.rolling(time=avg_n, center=True).mean().plot.line(x='time', label="Baseline", ax=axs[i, j]) #
                axs[i, j].set_title(f"WD ={wd}, Turbine {i}")
                
                x_start = data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(turb=0).ws_a.rolling(time=avg_n, center=True).mean().dropna('time').time.values.min()
                x_end = data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(turb=0).ws_a.rolling(time=avg_n, center=True).mean().dropna('time').time.values.max()

                axs[i, j].grid()
                axs[i, j].set_xlim(x_start, x_end)
                axs[i, j].set_ylabel(" ")
                axs[i, j].set_xlabel(" ")
                
        fig.supylabel("Wind speed [m/s]", fontsize=15, fontweight='bold')
        fig.supxlabel("Time [s]", fontsize=15, fontweight='bold')
        fig.suptitle(f"Rotor wind speed, ws={ws}, WD = {WDS}, TI = {TI}, TurbBox = {TURBBOX}", fontsize=15, fontweight='bold')
        axs[0,0].legend()
        plt.tight_layout()
        if save:
            plt.savefig(self.name + "_yaw_farm.png")
        return fig, axs
    
    def plot_turb(self, ws, wd, avg_n=10, TI=0.07, TURBBOX="Default", axs=None, save=False):
        data = self.multiple_eval_ds #Just for easier writing
        n_turb = len(data.turb.values)  #The number of turbines in the farm
        # n_wds = len(WDS)  #The number of wind directions we are looking at

        plot_x = ["Power", "Yaw", "Rotor wind speed"]
        
        if axs is None:
            fig, axs = plt.subplots(n_turb, len(plot_x), figsize=(18, 9), sharex=True)
        else:
            fig = axs[0,0].get_figure()

        for i in range(n_turb):
            #Bookkeeping for the different variables
            for j, plot_var in enumerate(plot_x):
                if plot_var == "Power":
                    to_plot = "powerT_"
                    plot_title = "Turbine power"
                    y_label = "Power [W]"
                elif plot_var == "Yaw":
                    to_plot = "yaw_"
                    plot_title = "Yaw offset [deg]"
                    y_label = "Yaw offset [deg]"
                elif plot_var == "Rotor wind speed":
                    to_plot = "ws_"
                    plot_title = "Rotor wind speed [m/s]"

                #Set the y axis to be shared between the different plots
                axs[i,j].sharey(axs[0,j])

                #Plot the data
                data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(turb=i).data_vars[to_plot+"a"].rolling(time=avg_n, center=True).mean().dropna("time").plot.line(x='time', label='Agent', ax=axs[i, j])
                data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(turb=i).data_vars[to_plot+"b"].rolling(time=avg_n, center=True).mean().dropna("time").plot.line(x='time', label='Baseline', ax=axs[i, j])
                
                #Set the title of the plot
                if i == 0:
                    axs[i, j].set_title(plot_title)
                else:
                    axs[i, j].set_title("")

                #Find at set the x-axis limits
                x_start = data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(turb=i).data_vars[to_plot+"a"].rolling(time=avg_n, center=True).mean().dropna("time").time.values.min()
                x_end = data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(turb=i).data_vars[to_plot+"a"].rolling(time=avg_n, center=True).mean().dropna("time").time.values.max()
                axs[i, j].set_xlim(x_start, x_end)

                #Set the y and x labels
                if j == 0:
                    axs[i, j].set_ylabel(f"Turbine {i}")
                else:
                    axs[i, j].set_ylabel(" ")
                if i == n_turb-1:
                    axs[i, j].set_xlabel("Time [s]")
                else:
                    axs[i, j].set_xlabel(" ")
                    

        fig.suptitle(f"Turbine power, yaw and rotor windspeed, ws={ws}, WD = {wd}, TI = {TI}, TurbBox = {TURBBOX}", fontsize=15, fontweight='bold')
        axs[0,0].legend()
        plt.tight_layout()
        plt.show()
        if save:
            plt.savefig(self.name + "_turbine_metrics.png")
        return fig, axs