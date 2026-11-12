import numpy as np
import pandas as pd
from .green_times_function import update_green_times
import shutil

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
            merged_df=pd.merge(df_sim[["edge_id","traveltime"]],self.df_gt[["edge_id","traveltime_mean"]],how="inner",on="edge_id",suffixes=["_sim","_gt"])
            mse=np.square(merged_df["traveltime_sim"]-merged_df["traveltime_mean_gt"]).mean()
            mse_list.append(mse)
        return np.mean(mse_list)