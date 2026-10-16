#"./Ingolstadt Network/additional_gt.xml"

from lxml import etree
import pandas as pd
import numpy as np


def create_initial_tl_df(gt_additional_xml):

    tree = etree.parse(gt_additional_xml)
    rows = [
        {
            "tl": phase.getparent().get("id"),
            "state":phase.attrib["state"],
            "duration":phase.attrib["duration"]
        }
        for phase in tree.xpath(".//tlLogic/phase")
    ]

    initial_tl_df = pd.DataFrame(rows)
    initial_tl_df["duration"]=initial_tl_df["duration"].astype(int)

    initial_tl_df_max_duration=initial_tl_df.groupby("tl")["duration"].max()
    initial_tl_df_second_max_duration = (
        initial_tl_df.groupby("tl")["duration"]
        .agg(lambda x: x.nlargest(2).iloc[-1])
    )
    #initial_tl_df["change"]=initial_tl_df["duration"].map()


    initial_tl_df["tl_max_duration"]=initial_tl_df["tl"].map(initial_tl_df_max_duration)
    initial_tl_df["tl_second_max_duration"]=initial_tl_df["tl"].map(initial_tl_df_second_max_duration)

    initial_tl_df["to_change"] = 0

    col_idx = initial_tl_df.columns.get_loc("to_change")

    for i in range(len(initial_tl_df)):
        tl = initial_tl_df.iloc[i]["tl"]
        duration = initial_tl_df.iloc[i]["duration"]
        tl_max_duration = initial_tl_df.iloc[i]["tl_max_duration"]
        tl_second_max_duration = initial_tl_df.iloc[i]["tl_second_max_duration"]

        tl_labels = initial_tl_df.loc[
            initial_tl_df["tl"] == tl, "to_change"
        ]

        if duration == tl_max_duration and 1 not in tl_labels.values:
            initial_tl_df.iloc[i, col_idx] = 1

        elif duration == tl_second_max_duration and 2 not in tl_labels.values:
            initial_tl_df.iloc[i, col_idx] = 2


    initial_tl_df=initial_tl_df.rename(columns={"duration":"initial_duration"})
    return initial_tl_df