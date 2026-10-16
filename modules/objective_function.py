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
                 tl_ids
                 ):
        
        self.initial_tl_df=initial_tl_df
        self.output_dir=output_dir
        self.sumo_simulator=sumo_simulator
        self.df_gt=df_gt
        self.additional_xml=additional_xml
        self.tl_ids=tl_ids

    def compute_error(self,g_vec,epoch_index):
        update_green_times(g_vec,self.additional_xml,self.tl_ids,self.initial_tl_df)
        shutil.copy(self.additional_xml, f'{self.output_dir}{epoch_index}_copy_additional.xml')
        self.sumo_simulator.simulation(epoch_index)
        df_sim=pd.read_csv(f"{self.output_dir}{epoch_index}_InTAS.travel_times.csv")
        merged_df=pd.merge(df_sim[["edge_id","traveltime_mean"]],self.df_gt[["edge_id","traveltime_mean"]],how="inner",on="edge_id",suffixes=["_sim","_gt"])
        mse=np.square(merged_df["traveltime_mean_sim"]-merged_df["traveltime_mean_gt"]).mean()
        return mse