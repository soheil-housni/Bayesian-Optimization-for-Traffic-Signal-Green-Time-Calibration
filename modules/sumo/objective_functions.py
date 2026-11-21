import numpy as np
import pandas as pd
from .green_times_function import update_green_times
import shutil
from .vehroute_df_preprocessing import create_vehroute_df
from .extract_route_average_traveltime import get_route_average_traveltime
from .sumo_simulation import SumoSimulation
from concurrent.futures import ProcessPoolExecutor
import time



def run_single_rep_error(rep,
                         epoch_index,
                         cfg_file,
                         output_dir,
                         tl_ids,
                         corridors_df,
                         df_gt,
                         demand_scaling_factor=None):

       time.sleep(rep * 1.0)
       
       sumo_simulator = SumoSimulation(
        cfg_file=cfg_file,
        output_dir=output_dir,
        tl_ids=tl_ids
        )
       sumo_simulator.simulation(epoch_index=epoch_index, rep=rep,demand_scaling_factor=demand_scaling_factor)

       vehroute_df=create_vehroute_df(f"Ingolstadt Network/{output_dir}{epoch_index}_{rep}_InTAS.vehroute.xml")

       corridors_df_sim=corridors_df.copy()

       corridors_df_sim.to_csv(f"./{output_dir}{epoch_index}_{rep}_corridors_df_copy.csv")

       corridors_df_sim["route_average_traveltime"]=corridors_df_sim.apply(get_route_average_traveltime,args=(vehroute_df,),axis=1)
       merged_df=pd.merge(df_gt[["route_id","route_average_traveltime","route_weight"]],corridors_df_sim[["route_id","route_average_traveltime"]],suffixes=["_gt","_sim"],on="route_id",how="inner")
       merged_df=merged_df.dropna(subset=["route_average_traveltime_sim","route_average_traveltime_gt"],axis=0, how="any")
       mse=(merged_df["route_weight"]*np.square(
            merged_df["route_average_traveltime_sim"]-merged_df["route_average_traveltime_gt"]
            )).sum()/merged_df["route_weight"].sum()
       average_tt=merged_df["route_average_traveltime_sim"].mean()

       return mse,average_tt


class ErrorFunctionRouteVersion():
    def __init__(self,
                 initial_tl_df,
                 df_gt,
                 sumo_simulator,
                 corridors_df,
                 additional_xml,
                 output_dir,
                 tl_ids):

        self.initial_tl_df=initial_tl_df
        self.sumo_simulator=sumo_simulator
        self.additional_xml=additional_xml
        self.output_dir=output_dir
        self.tl_ids=tl_ids
        self.corridors_df=corridors_df.sort_values(by="route_id")
        self.df_gt=df_gt.sort_values(by="route_id")
        self.df_gt["route_weight"]=self.df_gt["route_weight"]/self.df_gt["route_weight"].sum()

        


    def compute_error(self,
                      g_vec,
                      epoch_index,
                      num_replications,
                      travel_time=False,
                      demand_scaling_factor=None,
                      num_workers=5):

        update_green_times(g_vec,self.additional_xml,self.tl_ids,self.initial_tl_df)
        shutil.copy(self.additional_xml, f'{self.output_dir}{epoch_index}_copy_additional.xml')

        args_list = [
            (
                rep,
                epoch_index,
                self.sumo_simulator.cfg_file,
                self.sumo_simulator.output_dir,
                self.tl_ids,
                self.corridors_df,
                self.df_gt,
                demand_scaling_factor
            )
            for rep in range(num_replications)
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(run_single_rep_error, *args) for args in args_list]

            results = [future.result() for future in futures]
             
        mse_list = [result[0] for result in results]
        tt_list = [result[1] for result in results]
        

        if travel_time:
            return (np.mean(mse_list),np.mean(tt_list))
        else:
            return np.mean(mse_list) 
