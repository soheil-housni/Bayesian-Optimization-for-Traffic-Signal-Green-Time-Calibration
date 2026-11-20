import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np



def create_vehroute_df(vehroute_file):
    tree = ET.parse(vehroute_file)
    root = tree.getroot()

    data = []

    for vehicle in root.findall("vehicle"):
        vehicle_id = vehicle.get("id")
        depart_time=vehicle.get("depart")

        for route in vehicle.findall("route"):
            data.append({
                "vehicle_id": vehicle_id,
                "depart":vehicle.get("depart"),
                "route": route.get("edges").split(" "),
                "exit_times": [float(x) for x in route.get("exitTimes").split(" ")]
            })

    df = pd.DataFrame(data)

    df["route_str"]=df["route"].astype(str)
    df["route_str_str"]=df["route"].str.join(" ")
    filtered_df=df[~df["exit_times"].apply(lambda x : -1 in x)]
    nb_excluded_vehicles=len(df)-len(filtered_df)
    print("NB of excluded vehicles :")
    print(nb_excluded_vehicles)

    return filtered_df
