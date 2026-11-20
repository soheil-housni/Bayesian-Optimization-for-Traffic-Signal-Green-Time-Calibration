from .vehroute_df_preprocessing import create_vehroute_df
from .extract_route_connexion_traveltime import get_route_connexion_average_traveltime
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import time
from .sumo_simulation import SumoSimulation


def run_single_simulation(
                          epoch_index,
                          rep,
                          cfg_file,
                          output_dir,
                          tl_ids):
    time.sleep(rep*1)
    gt_sumo_simulator=SumoSimulation(cfg_file=cfg_file,output_dir=output_dir,tl_ids=tl_ids)
    gt_sumo_simulator.simulation(epoch_index=epoch_index,rep=rep)

def create_gt_dataset(
        complete_corridors_connexions_df,
        output_dir,
        epoch_index,
        num_replications,
        cfg_file,
        tl_ids,
        num_workers=5
):
    df_gt=None
    
    args_list=[(
                epoch_index,
                rep,
                cfg_file,
                output_dir,
                tl_ids) for rep in range(num_replications)]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_single_simulation, *args) for args in args_list]
        for f in futures:
            f.result()  

    for rep in range(num_replications):
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
    