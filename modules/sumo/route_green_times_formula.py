import pandas as pd
import numpy as np

def get_route_gt(g_vec,corridors_df,initial_tl_df):
    corridors_df=corridors_df.copy()
    g_vec_dict=create_g_vec_dict(initial_tl_df,g_vec)
    corridors_df["g_value"]=corridors_df["tl"].map(g_vec_dict)
    corridors_df["gt"]=corridors_df.apply(get_route_gt_formula,axis=1,args=(initial_tl_df,))
    return corridors_df


def create_g_vec_dict(initial_tl_df,g_vec):
    g_vec_dict={}
    tl_ids=np.sort(initial_tl_df["tl"].unique())
    if len(tl_ids)!=len(g_vec):
        raise ValueError("the number of tl ids should be equal to the dimension of the green vector")
    for i in range(len(tl_ids)):
        g_vec_dict[tl_ids[i]]=g_vec[i]

    return g_vec_dict

def get_route_gt_formula(row,initial_tl_df):
    g=row["g_value"]
    main_green_phase=row["main_green_phase"]
    tl=row["tl"]
    tl_cumulative_constant_duration=row["tl_cumulative_constant_duration"]
    filtered_initial_tl_df=initial_tl_df.loc[(initial_tl_df["tl"]==tl) & (initial_tl_df["state"]==main_green_phase)]
    if filtered_initial_tl_df["to_change"].iloc[0]==0:
        return filtered_initial_tl_df["initial_duration"].iloc[0]
    elif filtered_initial_tl_df["to_change"].iloc[0]==1:
        return g
    elif filtered_initial_tl_df["to_change"].iloc[0]==2:
        return 90-g-tl_cumulative_constant_duration
    else:
        return None



