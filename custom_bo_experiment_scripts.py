import numpy as np
import pandas as pd
from modules.bo import optimize_acqf_and_get_observation,BOLoop,InitialSamples
from modules.msr import raasp_ic_generator,RAASP_sampler
from modules.sumo import create_initial_tl_df,check_feasibility,update_green_times,ErrorFunctionRouteVersion,SumoSimulation,get_upper_bounds
import sumolib
from lxml import etree
import torch
import sympy as sp
from torch.quasirandom import SobolEngine
from modules.turbo import CustomThompsonSampler,tr_gp_fit,TurboLoop,TrustRegionInitialization,TRCandidatesGeneration
import argparse
from itertools import product
import os
import shutil
import random
import ast
from modules.gp import InitaliseGPModel,UpdatedFAFunctionLinearRegresion

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_sumo_files(method,annotation,rs,kernel_name="MaternKernel_default",mean_name="ConstantMean"):

    set_seed(rs)

    network_dir = "./Ingolstadt Network"

    original_additional = os.path.join(
        network_dir, "additional.xml"
    )

    original_config = os.path.join(
        network_dir, "InTAS.sumocfg"
    )

    additional_xml = os.path.join(
        network_dir, f"additional_{method}_{kernel_name}_{mean_name}_{annotation}_rs{rs}.xml"
    )

    cfg_file = os.path.join(
        network_dir, f"InTAS_{method}_{kernel_name}_{mean_name}_{annotation}_rs{rs}.sumocfg"
    )

    shutil.copy2(original_additional, additional_xml)
    shutil.copy2(original_config, cfg_file)

    tree = etree.parse(cfg_file)

    additional_files = tree.find(".//additional-files")

    if additional_files is None:
        raise ValueError("additional-files introuvable dans le sumocfg")

    additional_files.set("value", f"additional_{method}_{kernel_name}_{mean_name}_{annotation}_rs{rs}.xml")

    tree.write(cfg_file, encoding="UTF-8", xml_declaration=True)

    return cfg_file, additional_xml


def run_vanilla(method,id_init_sample,initial_length_scale_value,rs,kernel_name="MaternKernel_default",mean_name="ConstantMean",scaled=False):
    set_seed(rs)

    niter=30
    init_df=torch.load(f"./initial_samples/init_df_{id_init_sample}.pt",weights_only=False)
    output_dir="BO_output/"
    dtype=torch.double
    x_dim=98
    lower_bounds=[5]*x_dim
    additional_xml_gt="./Ingolstadt Network/additional_gt.xml"
    initial_tl_df=create_initial_tl_df(additional_xml_gt)
    upper_bounds=get_upper_bounds(initial_tl_df)
    bounds=torch.tensor(
        [lower_bounds,
        upper_bounds],
        dtype=dtype
    )

    if scaled:
            annotation=f"ls_{initial_length_scale_value}scaled_is{id_init_sample}"
            initial_length_scale_value=initial_length_scale_value*np.sqrt(x_dim)
    else:
        annotation=f"ls_{initial_length_scale_value}_is{id_init_sample}"


    df_gt=pd.read_csv("./gt_data/df_gt_InTAS.csv")
    df_gt["route"]=df_gt["route"].apply(ast.literal_eval)
    net=sumolib.net.readNet("Ingolstadt Network/ingolstadt.net.xml")
    tl_ids=[tl.getID() for tl in net.getTrafficLights()]
    num_replications=5

    cfg_file, additional_xml = prepare_sumo_files(method=method,annotation=annotation,rs=rs,kernel_name=kernel_name,mean_name=mean_name)

    raw_samples=min(100*x_dim,5000)
    batch_size=1
    num_restarts_opti=5
    device=(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))


    sumo_simulator=SumoSimulation(cfg_file=cfg_file,
                                    output_dir=output_dir,
                                    tl_ids=tl_ids,
                                    )

    corridors_df=torch.load("./corridors_savings/corridors_df.pt",weights_only=False)
    
    objective_function=ErrorFunctionRouteVersion(
        initial_tl_df=initial_tl_df,
        output_dir=output_dir,
        sumo_simulator=sumo_simulator,
        df_gt=df_gt,
        additional_xml=additional_xml,
        tl_ids=tl_ids,
        corridors_df=corridors_df
    )

    net_file="Ingolstadt Network/ingolstadt.net.xml"

    fA_function=UpdatedFAFunctionLinearRegresion(
         df_gt=df_gt,
         initial_tl_df=initial_tl_df,
         net_file=net_file
    )

    gp_initializer=InitaliseGPModel(
         x_bounds=bounds,
         fA_function=fA_function
    )

    BO_instance=BOLoop(niter=niter,
                    init_df=init_df,
                    output_dir=output_dir,
                    bounds=bounds,
                    objective_function=objective_function,
                    dtype=dtype,
                    x_dim=x_dim,
                    device=device,
                    num_restarts_opti=num_restarts_opti,
                    raw_samples=raw_samples,
                    batch_size=batch_size,
                    initial_tl_df=initial_tl_df,
                    gp_initializer=gp_initializer
                    )


    BO_instance.run_loop(version="VANILLA",
                         initial_length_scale_value=initial_length_scale_value,
                         rs=rs,
                         annotation=annotation,
                         num_replications=num_replications,
                         kernel_name=kernel_name,
                         mean_name=mean_name)



