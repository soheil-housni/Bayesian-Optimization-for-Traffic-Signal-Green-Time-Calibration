import numpy as np
import pandas as pd
from modules.bo import optimize_acqf_and_get_observation,BOLoop,initialize_gp_model,InitialSamples
from modules.msr import raasp_ic_generator,RAASP_sampler
from modules.sumo import create_initial_tl_df,check_feasibility,update_green_times,ErrorFunction,SumoSimulation,get_upper_bounds
import sumolib
from lxml import etree
import torch
import sympy as sp
from torch.quasirandom import SobolEngine
from modules.turbo import CustomThompsonSampler,tr_gp_fit,TurboLoop,TrustRegionInitialization,TRCandidatesGeneration
import argparse
import os
import shutil
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Pour favoriser la reproductibilité
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_sumo_files(id_init_sample):

    network_dir = "./Ingolstadt Network"

    original_additional = os.path.join(
        network_dir, "additional.xml"
    )

    original_config = os.path.join(
        network_dir, "InTAS.sumocfg"
    )

    # Fichiers propres à cet initial sample
    additional_xml = os.path.join(
        network_dir, f"additional_init_{id_init_sample}.xml"
    )

    cfg_file = os.path.join(
        network_dir, f"InTAS_init_{id_init_sample}.sumocfg"
    )

    # Copier les fichiers originaux
    shutil.copy2(original_additional, additional_xml)
    shutil.copy2(original_config, cfg_file)

    # Modifier la configuration pour référencer le XML spécifique
    tree = etree.parse(cfg_file)

    additional_files = tree.find(".//additional-files")

    if additional_files is None:
        raise ValueError("additional-files introuvable dans le sumocfg")

    additional_files.set("value", f"additional_init_{id_init_sample}.xml")

    tree.write(cfg_file, encoding="UTF-8", xml_declaration=True)

    return cfg_file, additional_xml



def generate_initial_samples(id_init_sample):
    set_seed(id_init_sample)

    df_gt=pd.read_csv("./gt_data/gt_InTAS.travel_times.csv")
    net=sumolib.net.readNet("Ingolstadt Network/ingolstadt.net.xml")
    tl_ids=[tl.getID() for tl in net.getTrafficLights()]

    additional_xml_gt="./Ingolstadt Network/additional_gt.xml"
    initial_tl_df=create_initial_tl_df(additional_xml_gt)
    num_replications=3
    output_dir_init="initial_output/"
    dtype=torch.double

    cfg_file, additional_xml = prepare_sumo_files(id_init_sample)

    init_sumo_simulator=SumoSimulation(num_replications=num_replications,
                                    cfg_file=cfg_file,
                                    output_dir=output_dir_init,
                                    tl_ids=tl_ids,
                                    )

    init_objective_function=ErrorFunction(
        initial_tl_df=initial_tl_df,
        output_dir=output_dir_init,
        sumo_simulator=init_sumo_simulator,
        df_gt=df_gt,
        additional_xml=additional_xml,
        tl_ids=tl_ids
    )

    initial_sampler=InitialSamples(dtype=dtype,
                                objective_function=init_objective_function,
                                initial_tl_df=initial_tl_df)

    sample_size=15
    x_dim=98
    lower_bounds=[5]*x_dim
    upper_bounds=get_upper_bounds(initial_tl_df)
    x_bounds=torch.tensor(
        [lower_bounds,
        upper_bounds],
        dtype=dtype
    )

    
    init_X_norm, init_Y, init_df=initial_sampler.get_initial_sample(sample_size=sample_size,x_dim=x_dim,x_bounds=x_bounds,id_init_sample=id_init_sample,seed=id_init_sample)


    torch.save(init_X_norm,f"./initial_samples/init_X_norm_{id_init_sample}.pt")
    torch.save(init_Y,f"./initial_samples/init_Y_{id_init_sample}.pt")
    torch.save(init_df,f"./initial_samples/init_df_{id_init_sample}.pt")



if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
                "--id_init_sample",
                type=int,
                required=False
    )


    args = parser.parse_args()


    generate_initial_samples(id_init_sample=args.id_init_sample)