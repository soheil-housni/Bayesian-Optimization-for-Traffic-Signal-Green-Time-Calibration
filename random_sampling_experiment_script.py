import pandas as pd
import numpy as np
from modules.random_sampling import RandomSampling
from modules.sumo import TLIncomingCorridors,GetRouteGreenTime,create_initial_tl_df,get_upper_bounds,ErrorFunctionRouteVersion,SumoSimulation
import sumolib
import os
import shutil
from lxml import etree
import torch
import random
import argparse


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_sumo_files(rs):
    set_seed(rs)
    network_dir = "./Ingolstadt Network"

    original_additional = os.path.join(
        network_dir, "additional.xml"
    )

    original_config = os.path.join(
        network_dir, "InTAS.sumocfg"
    )

    additional_xml = os.path.join(
        network_dir, f"additional_inverse_cdf_rs{rs}.xml"
    )

    cfg_file = os.path.join(
        network_dir, f"InTAS_inverse_cdf_rs{rs}.sumocfg"
    )

    shutil.copy2(original_additional, additional_xml)
    shutil.copy2(original_config, cfg_file)

    tree = etree.parse(cfg_file)

    additional_files = tree.find(".//additional-files")

    if additional_files is None:
        raise ValueError("additional-files introuvable dans le sumocfg")

    additional_files.set("value", f"additional_inverse_cdf_rs{rs}.xml")

    tree.write(cfg_file, encoding="UTF-8", xml_declaration=True)

    return cfg_file, additional_xml


def run_inverse_cdf(rs):
    set_seed(rs)
    net_xml="./Ingolstadt Network/ingolstadt.net.xml"

    additional_xml_gt="./Ingolstadt Network/additional_gt.xml"
    initial_tl_df=create_initial_tl_df(additional_xml_gt)
    df_gt=pd.read_csv("./gt_data/df_gt_InTAS.csv")
    x_dim=98
    lower_bounds=[5]*x_dim
    upper_bounds=get_upper_bounds(initial_tl_df)
    dtype=torch.double
    bounds=torch.tensor(
            [lower_bounds,
            upper_bounds],
            dtype=dtype
    )

    net=sumolib.net.readNet("Ingolstadt Network/ingolstadt.net.xml")
    tl_ids=[tl.getID() for tl in net.getTrafficLights()]


    cfg_file, additional_xml = prepare_sumo_files(rs=rs)


    sumo_simulator=SumoSimulation(cfg_file=cfg_file,
                                  output_dir=output_dir,
                                  tl_ids=tl_ids,
                                    )

    corridors_df=torch.load("./corridors_savings/corridors_df.pt",weights_only=False)

    output_dir="random_sampling_output/"
    objective_function=ErrorFunctionRouteVersion(
            initial_tl_df=initial_tl_df,
            output_dir=output_dir,
            sumo_simulator=sumo_simulator,
            df_gt=df_gt,
            additional_xml=additional_xml,
            tl_ids=tl_ids,
            corridors_df=corridors_df
        )
    niter=45
    num_replications=5
    random_sampling_loop=RandomSampling(objective_function=objective_function,
                                        niter=niter,
                                        num_replications=num_replications,
                                        x_bounds=bounds,
                                        initial_tl_df=initial_tl_df,
                                        output_dir=output_dir
                                        )
    random_sampling_loop.run_loop(rs=rs)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
                    "--rs",
                    type=int,
                    required=False
    )
    args = parser.parse_args()
    run_inverse_cdf(rs=args.rs)
    