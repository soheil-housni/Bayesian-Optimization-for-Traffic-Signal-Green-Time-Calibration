from .routes_green_times import GetRouteGreenTime
import torch
import pandas as pd
import numpy as np

class ExtractRoutesGreenTimes():
    def __init__(self,
                 df_gt,
                 net_file,
                 initial_tl_df,
                 dtype=torch.double):

        self.df_gt=df_gt
        self.initial_tl_df=initial_tl_df
        self.initial_tl_df=self.initial_tl_df.sort_values(by=["tl","state"])
        
        route_green_time_accessor=GetRouteGreenTime(net_file=net_file,initial_tl_df=initial_tl_df)
        self.df_gt=route_green_time_accessor.get_route_green_time(self.df_gt)
        self.df_gt=pd.merge(self.df_gt,self.initial_tl_df[["tl","state","tl_cumulative_constant_duration"]].rename(columns={"state":"main_green_phase"}),on=["tl","main_green_phase"],how="left")
        self.df_gt=self.df_gt.drop_duplicates(subset=["route_str_str"])

        self.df_gt=self.df_gt.sort_values(by=["route_id"])
        self.df_gt=self.df_gt.reset_index(drop=True)

        self.tl_ids=np.sort(initial_tl_df["tl"].unique())

        self.routes_to_vary=self.df_gt[self.df_gt["main_green_phase_to_change"]==1].index
        self.routes_to_change_index=self.df_gt[self.df_gt["main_green_phase_to_change"]==2].index
        self.routes_constant_index=self.df_gt[self.df_gt["main_green_phase_to_change"]==0].index

        self.routes_constant_values=self.df_gt["main_green_phase_initial_duration"].values
        self.route_tl_cumulative_constant_values=self.df_gt["tl_cumulative_constant_duration"].values


        self.tl_route_correspondance_df,self.tensor_matrix=self.create_tl_route_correspondance_matrix()

        self.dtype=dtype



    def get_routes_green_times(self,x_init):


        route_green_time=torch.matmul(x_init,self.tensor_matrix)

        g_vec_delta=torch.tensor(90-self.route_tl_cumulative_constant_values,dtype=self.dtype).unsqueeze(0)
        route_green_time_change=g_vec_delta-route_green_time

        route_green_time_constant=torch.tensor(self.routes_constant_values,dtype=self.dtype).unsqueeze(0)

        route_green_time[...,self.routes_to_vary]=route_green_time[...,self.routes_to_vary]
        route_green_time[...,self.routes_to_change_index]=route_green_time_change[...,self.routes_to_change_index]
        route_green_time[...,self.routes_constant_index]=route_green_time_constant[...,self.routes_constant_index]

        return route_green_time



    def create_tl_route_correspondance_matrix(self):
        values=np.zeros((len(self.tl_ids),len(self.df_gt)))
        index=np.sort(self.tl_ids)
        columns=np.sort(self.df_gt["route_id"].values)
        tl_route_correspondance_df=pd.DataFrame(values,index=index,columns=columns)
        for route_id in columns:
            df_gt_filtered=self.df_gt[self.df_gt["route_id"]==route_id]
            tl=df_gt_filtered["tl"]
            tl_route_correspondance_df.loc[tl,route_id]=1

        tensor_matrix=torch.tensor(tl_route_correspondance_df.values,dtype=torch.double)
        return tl_route_correspondance_df,tensor_matrix
                
                


