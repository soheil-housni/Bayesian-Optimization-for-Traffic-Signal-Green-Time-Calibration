import numpy as np
import pandas as pd
import torch


class GreenTimeMapping():
    def __init__(self,
                df_gt,
                initial_tl_df):
        
        self.df_gt=df_gt
        self.initial_tl_df=initial_tl_df
        self.initial_tl_df=self.initial_tl_df.sort_values(by=["tl","state"])

    def compute_connexions_green_times(self,
                                       g_vec):
         
         np_array,all_connexion_phases_df=self.creation_route_phase_correspondance_matrix()
         complete_g_vec=self.get_complete_g_vec(g_vec)
         return np.matmul(complete_g_vec,np_array)
 
    def get_complete_g_vec(self,g_vec):
         index_to_vary,index_to_change,index_constant,constant_array=self.get_indices()
         complete_g_vec=np.zeros((len(self.initial_tl_df)))
         complete_g_vec[index_to_vary]=g_vec
         complete_g_vec[index_constant]=self.initial_tl_df.iloc[index_constant]["duration"].values
         complete_g_vec[index_to_change]=90-g_vec-constant_array

         return complete_g_vec
         


    def creation_route_phase_correspondance_matrix(self):
            
            self.initial_tl_df["tl_id_state"]=self.initial_tl_df.apply(lambda x : (x["tl"],x["state"]), axis=1)
            phases=self.initial_tl_df["tl_id_state"].values
            index = pd.MultiIndex.from_tuples(phases, names=["tl_id", "state"])
            connexion_index=np.sort(self.df_gt["connexion_id"].values)
            connexion_phases_df=pd.DataFrame(np.zeros((len(self.complete_corridors_connexions_df), len(phases))),columns=index,index=connexion_index)

            for i in connexion_index:
                tl=self.df_gt.loc[self.df_gt["connexion_id"]==i,"tl"].iloc[0]
                green_phases=self.df_gt.loc[self.df_gt["connexion_id"]==i,"green_phases"].iloc[0]
                for green_phase in green_phases:
                    connexion_phases_df.loc[i,(tl,green_phase)]=1

            connexion_phases_df.columns=connexion_phases_df.columns.to_flat_index()
            connexion_phases_df=connexion_phases_df.reset_index().rename(columns={"index":"connexion_id"})
            all_connexion_phases_df=pd.merge(self.df_gt,connexion_phases_df,on="connexion_id",how="inner")
            np_array=all_connexion_phases_df[phases].values
            return np_array,all_connexion_phases_df


    def get_indices(self):
        tl_df=self.initial_tl_df.copy()
        constant_list=[]
        index_to_vary=tl_df[tl_df["to_change"]==1].index()
        index_to_change=tl_df[tl_df["to_change"]==2].index()
        index_constant=tl_df[tl_df["to_change"]==0].index()
        for tl in tl_df["tl"].unique():
             filtered_tl_df=tl_df[(tl_df["tl"]==tl) & (tl_df["to_change"]==0)]
             constant_list.append(filtered_tl_df)

        return index_to_vary,index_to_change,index_constant,np.array(constant_list)
         