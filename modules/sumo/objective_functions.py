import numpy as np
import pandas as pd
from .green_times_function import update_green_times
import shutil
from .vehroute_df_preprocessing import create_vehroute_df
from .extract_route_connexion_traveltime import get_route_connexion_average_traveltime

class ErrorFunction():
    def __init__(self,
                 initial_tl_df,
                 output_dir,
                 sumo_simulator,
                 df_gt,
                 additional_xml,
                 tl_ids,
                 ):
        
        self.initial_tl_df=initial_tl_df
        self.output_dir=output_dir
        self.sumo_simulator=sumo_simulator
        self.df_gt=df_gt
        self.df_gt=self.df_gt.rename(columns={"traveltime_mean":"traveltime_mean_gt"})
        self.additional_xml=additional_xml
        self.tl_ids=tl_ids

    def compute_error(self,g_vec,epoch_index,num_replications,demand_scaling_factor=None):
        update_green_times(g_vec,self.additional_xml,self.tl_ids,self.initial_tl_df)
        shutil.copy(self.additional_xml, f'{self.output_dir}{epoch_index}_copy_additional.xml')
        gt_edge_ids_set=self.df_gt["edge_id"].unique()
        mse_list=[]
        for rep in range(num_replications):
            self.sumo_simulator.simulation(epoch_index=epoch_index,rep=rep,demand_scaling_factor=demand_scaling_factor)
            df_sim=pd.read_csv(f"{self.output_dir}{epoch_index}_{rep}_InTAS.travel_times.csv")
            df_sim=df_sim.loc[df_sim["edge_id"].isin(gt_edge_ids_set)]
            df_sim=df_sim.rename(columns={"traveltime":"traveltime_sim"})
            merged_df=pd.merge(df_sim[["edge_id","traveltime_sim"]],self.df_gt[["edge_id","traveltime_mean_gt"]],how="inner",on="edge_id",suffixes=["_sim","_gt"])
            mse=np.square(merged_df["traveltime_sim"]-merged_df["traveltime_mean_gt"]).mean()
            mse_list.append(mse)
        return np.mean(mse_list)





class ErrorFunctionConnexionVersion():
    def __init__(self,
                 initial_tl_df,
                 df_gt,
                 sumo_simulator,
                 complete_corridors_connexions_df,
                 additional_xml,
                 output_dir,
                 tl_ids):

        self.initial_tl_df=initial_tl_df
        self.sumo_simulator=sumo_simulator
        self.additional_xml=additional_xml
        self.output_dir=output_dir
        self.tl_ids=tl_ids


        self.complete_corridors_connexions_df=complete_corridors_connexions_df.sort_values(by="connexion_id")
        self.df_gt=df_gt=df_gt.sort_values(by="connexion_id")
        


    def compute_error(self,
                      g_vec,
                      epoch_index,
                      num_replications,
                      travel_time=False,
                      demand_scaling_factor=None):

        update_green_times(g_vec,self.additional_xml,self.tl_ids,self.initial_tl_df)
        shutil.copy(self.additional_xml, f'{self.output_dir}{epoch_index}_copy_additional.xml')
        mse_list=[]
        tt_list=[]

        for rep in range(num_replications):

            self.sumo_simulator.simulation(epoch_index=epoch_index, rep=rep,demand_scaling_factor=demand_scaling_factor)
            vehroute_df=create_vehroute_df(f"ingolstadt7/{self.output_dir}{epoch_index}_{rep}_InTAS7DIM.vehroute.xml")
            complete_corridors_connexions_df_sim=self.complete_corridors_connexions_df.copy()
            complete_corridors_connexions_df_sim["route_connexion_average_traveltime"]=complete_corridors_connexions_df_sim.apply(get_route_connexion_average_traveltime,args=(vehroute_df,),axis=1)
            merged_df=pd.merge(self.df_gt[["connexion_id","route_connexion_average_traveltime","connexion_weight"]],complete_corridors_connexions_df_sim[["connexion_id","route_connexion_average_traveltime"]],suffixes=["_gt","_sim"],on="connexion_id",how="inner")
            merged_df=merged_df.dropna(subset=["route_connexion_average_traveltime_sim","route_connexion_average_traveltime_gt"],axis=0, how="any")

            mse=(merged_df["connexion_weight"]*np.square(
                merged_df["route_connexion_average_traveltime_sim"]-merged_df["route_connexion_average_traveltime_gt"]
            )).sum()

            average_tt=merged_df["route_connexion_average_traveltime_sim"].mean()

            mse_list.append(mse)
            tt_list.append(average_tt)

        if travel_time:
            return (np.mean(mse_list),np.mean(tt_list))
        else:
            return np.mean(mse_list) 
