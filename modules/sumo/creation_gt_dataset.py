from .vehroute_df_preprocessing import create_vehroute_df
from .extract_route_average_traveltime import get_route_average_traveltime
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
        corridors_df,
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
        corridors_df_copy=corridors_df.copy()
        corridors_df_copy["route_average_traveltime"]=corridors_df_copy.apply(get_route_average_traveltime,args=(vehroute_df,),axis=1)
        corridors_df_copy=corridors_df_copy.dropna(subset=["route_average_traveltime"])

        corridors_df_copy.to_csv(f"./{output_dir}corridors_df_copy_{rep}.csv")

        if df_gt is None:
            df_gt=corridors_df_copy
        else:
            df_gt=pd.concat([df_gt,corridors_df_copy],axis=0)

    df_gt=df_gt.groupby("route_id").agg({"route_average_traveltime":"mean"})
    df_gt=pd.merge(df_gt,corridors_df,how="left",on="route_id")
    total_demand=df_gt["route_demand"].sum()
    df_gt["route_weight"]=df_gt["route_demand"]/total_demand

    df_gt.to_csv(f"./{output_dir}df_gt_InTAS.csv")
    