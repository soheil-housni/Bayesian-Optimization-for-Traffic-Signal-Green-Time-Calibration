import xml.etree.ElementTree as ET
import numpy as np

def update_green_times(g_vec,
                       additional_xml,
                       tl_ids_set,
                       initial_tl_df):

    # update green time dictionary
    tl_df_updated,g_vec_all = update_green_time_df(g_vec,initial_tl_df)
    
    # update additional.xml     
    tree = ET.parse(additional_xml)
    root = tree.getroot()
    
    # iterate across endogenous signalized intersections
    for tl in root.iter("tlLogic"):
        if tl.attrib["id"] in tl_ids_set:
            tl_curr = tl.attrib["id"]
            if tl.attrib["programID"] != "1":
                raise ValueError('We should only be changing the signal of programID 1')
            
            for ph in tl:
                ph_state=ph.attrib["state"]
                if not (tl_df_updated[(tl_df_updated["state"]==ph_state) & (tl_df_updated["tl"]==tl_curr)].empty):
                    ph.attrib["duration"] = str(int(tl_df_updated[(tl_df_updated["state"]==ph_state) & (tl_df_updated["tl"]==tl_curr)]["duration"].iloc[0]))
    tree.write(additional_xml)     
    print(f'Wrote {additional_xml}')
    

def update_green_time_df(g_vec,initial_tl_df):

    tl_df=initial_tl_df.copy()

    g_vec_rounded = np.floor(g_vec).astype(np.int64)
    tl_df=tl_df.sort_values(by=["tl","state"])
    tl_df["duration"]=tl_df["initial_duration"]

    rest_durations=tl_df.loc[tl_df["to_change"]==0].groupby("tl")["initial_duration"].sum().reset_index().sort_values(by=["tl"])
    rest_durations_values=rest_durations["initial_duration"].values

    tl_df.loc[tl_df["to_change"]==1,"duration"]=g_vec_rounded
    tl_df.loc[tl_df["to_change"]==2,"duration"]=90-g_vec_rounded-rest_durations_values

    tl_df_updated=tl_df[tl_df["to_change"]!=0]
    g_vec_all=tl_df_updated["duration"].values
    return tl_df_updated,g_vec_all