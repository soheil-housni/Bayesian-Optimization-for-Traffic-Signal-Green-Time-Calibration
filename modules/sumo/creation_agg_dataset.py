import pandas as pd
import numpy as np


def create_agg_dataset(output_dir,
                       epoch_index,
                       num_replications):
    df_all=None
    for rep in range(num_replications):
        df_sim=pd.read_csv(f"{output_dir}{epoch_index}_{rep}_InTAS.travel_times.csv")
        if df_all is None:
            df_all=df_sim
        else:
            df_all=pd.concat([df_all,df_sim],axis=0)

    edges_count=df_all.value_counts("edge_id").reset_index()
    df_all=pd.merge(df_all,edges_count,on="edge_id",how="left")
    df_all=df_all.loc[df_all["count"]==num_replications]


    df_all_agg=df_all.groupby("edge_id").agg({"sampledSeconds":['mean', 'std'],"traveltime":['mean', 'std'],\
                                                "density":['mean', 'std'],"occupancy":['mean', 'std'],"speed":['mean', 'std'],"waitingTime":['mean', 'std'],\
                                                "timeLoss":['mean', 'std'],"departed":['mean', 'std'],"arrived":['mean', 'std'],"entered":['mean', 'std'],"left":['mean', 'std'],\
                                                "flow":['mean', 'std']}).reset_index()
    df_all_agg=df_all_agg.droplevel(1,axis=1)
    new_column_names=["edge_id"]
    for i,col in enumerate(df_all_agg.columns[1:]):
        if i%2==0:
            new_column_names.append(col+"_mean")
        else:
            new_column_names.append(col+"_std")

    df_all_agg.columns=new_column_names
    df_all_agg.to_csv(f"{output_dir}{epoch_index}_InTAS.travel_times.csv")