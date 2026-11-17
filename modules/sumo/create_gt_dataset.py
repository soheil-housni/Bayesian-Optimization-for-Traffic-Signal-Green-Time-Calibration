from .vehroute_df_preprocessing import create_vehroute_df
from .extract_route_connexion_traveltime import get_route_connexion_average_traveltime
import pandas as pd

def create_gt_dataset(
        complete_corridors_connexions_df,
        output_dir,
        epoch_index,
        sumo_simulator,
        num_replications
):
    df_gt=None
    for rep in range(num_replications):
        sumo_simulator.simulation(epoch_index=epoch_index,rep=rep)
        vehroute_df=create_vehroute_df(f"Ingolstadt Network/{output_dir}{epoch_index}_{rep}_InTAS.vehroute.xml")
        complete_corridors_connexions_df_copy=complete_corridors_connexions_df.copy()
        complete_corridors_connexions_df_copy["route_connexion_average_traveltime"]=complete_corridors_connexions_df_copy.apply(get_route_connexion_average_traveltime,args=(vehroute_df,),axis=1)
        complete_corridors_connexions_df_copy=complete_corridors_connexions_df_copy.dropna(subset=["route_connexion_average_traveltime"])

        complete_corridors_connexions_df_copy.to_csv(f"./{output_dir}complete_corridors_connexions_df_copy_{rep}.csv")

        if df_gt is None:
            df_gt=complete_corridors_connexions_df_copy
        else:
            df_gt=pd.concat([df_gt,complete_corridors_connexions_df_copy],axis=0)

    df_gt=df_gt.groupby("connexion_id").agg({"route_connexion_average_traveltime":"mean"})
    df_gt=pd.merge(df_gt,complete_corridors_connexions_df,how="left",on="connexion_id")
    total_demand=df_gt["connexion_demand"].sum()
    df_gt["connexion_weight"]=df_gt["connexion_demand"]/total_demand

    df_gt.to_csv(f"./{output_dir}df_gt_InTAS.csv")
    