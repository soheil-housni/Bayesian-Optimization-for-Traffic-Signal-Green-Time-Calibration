import pandas as pd
import numpy as np



def get_route_average_traveltime(row,vehroute_df):
    route=row["route"]
    route_str_str=row["route_str_str"]

    vehroute_df_route=vehroute_df[vehroute_df["route_str_str"].str.contains(route_str_str,regex=False)]


    if vehroute_df_route.empty:
        return np.nan

    def extract_times_index(vehroute_df_route_row,sub_route):
        full_route=vehroute_df_route_row["route"]
        start_indices = [i for i in range(len(full_route) - len(sub_route) + 1) if full_route[i:i+len(sub_route)] == sub_route]

        
        end_indices = [start_indx + len(sub_route) - 1 for start_indx in start_indices]

        return (start_indices,end_indices)

    vehroute_df_route["indices"]=vehroute_df_route.apply(extract_times_index,args=(route,),axis=1)

    vehroute_df_route["start_indices"] = vehroute_df_route["indices"].str[0]
    vehroute_df_route["end_indices"] = vehroute_df_route["indices"].str[1]

    vehroute_df_route["route_depart_times"]=vehroute_df_route.apply(
        lambda x : 
        [float(x["exit_times"][start_idx-1]) if start_idx>0 else float(x["depart"]) for start_idx in x["start_indices"]],
        axis=1)

    vehroute_df_route["route_exit_times"]=vehroute_df_route.apply(
        lambda x : [float(x["exit_times"][end_indx]) for end_indx in x["end_indices"]],
        axis=1)

    vehroute_df_route["route_traveltime"]=vehroute_df_route.apply(
        lambda x : [exit_time-depart_time for exit_time,depart_time in zip(x["route_exit_times"],x["route_depart_times"])],
        axis=1
    )


    traveltime_values=[tt for tt_list in vehroute_df_route["route_traveltime"].values for tt in tt_list]
    average_route_traveltime=np.mean(traveltime_values)


    return average_route_traveltime