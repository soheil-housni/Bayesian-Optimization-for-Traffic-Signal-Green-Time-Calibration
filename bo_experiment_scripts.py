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
from itertools import product
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

def prepare_sumo_files(method,annotation,rs):

    set_seed(rs)

    network_dir = "./Ingolstadt Network"

    original_additional = os.path.join(
        network_dir, "additional.xml"
    )

    original_config = os.path.join(
        network_dir, "InTAS.sumocfg"
    )

    # Fichiers propres à cet initial sample
    additional_xml = os.path.join(
        network_dir, f"additional_{method}_{annotation}_rs{rs}.xml"
    )

    cfg_file = os.path.join(
        network_dir, f"InTAS_{method}_{annotation}_rs{rs}.sumocfg"
    )

    # Copier les fichiers originaux
    shutil.copy2(original_additional, additional_xml)
    shutil.copy2(original_config, cfg_file)

    # Modifier la configuration pour référencer le XML spécifique
    tree = etree.parse(cfg_file)

    additional_files = tree.find(".//additional-files")

    if additional_files is None:
        raise ValueError("additional-files introuvable dans le sumocfg")

    additional_files.set("value", f"additional_{method}_{annotation}_rs{rs}.xml")

    tree.write(cfg_file, encoding="UTF-8", xml_declaration=True)

    return cfg_file, additional_xml


def run_vanilla(method,id_init_sample,initial_length_scale_value,rs,scaled=False):
    set_seed(rs)

    niter=30
    num_restarts=3
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


    df_gt=pd.read_csv("./gt_data/gt_InTAS.travel_times.csv")
    net=sumolib.net.readNet("Ingolstadt Network/ingolstadt.net.xml")
    tl_ids=[tl.getID() for tl in net.getTrafficLights()]
    num_replications=3

    cfg_file, additional_xml = prepare_sumo_files(method=method,annotation=annotation,rs=rs)

    raw_samples=min(100*x_dim,5000)
    batch_size=1
    num_restarts_opti=5
    device=(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))


    sumo_simulator=SumoSimulation(num_replications=num_replications,
                                    cfg_file=cfg_file,
                                    output_dir=output_dir,
                                    tl_ids=tl_ids,
                                    )

    objective_function=ErrorFunction(
        initial_tl_df=initial_tl_df,
        output_dir=output_dir,
        sumo_simulator=sumo_simulator,
        df_gt=df_gt,
        additional_xml=additional_xml,
        tl_ids=tl_ids
    )

    BO_instance=BOLoop(niter=niter,
                    num_restart=num_restarts,
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
                    )


    BO_instance.run_loop(version="VANILLA",initial_length_scale_value=initial_length_scale_value,rs=rs,annotation=annotation)



def run_msr(method,id_init_sample,initial_length_scale_value,rs,scaled=True):
    set_seed(rs)

    niter=30
    num_restart=3
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


    df_gt=pd.read_csv("./gt_data/gt_InTAS.travel_times.csv")
    net=sumolib.net.readNet("Ingolstadt Network/ingolstadt.net.xml")
    tl_ids=[tl.getID() for tl in net.getTrafficLights()]
    num_replications=3

    if scaled:
            annotation=f"ls_{initial_length_scale_value}scaled_is{id_init_sample}"
            initial_length_scale_value=initial_length_scale_value*np.sqrt(x_dim)
    else:
        annotation=f"ls_{initial_length_scale_value}_is{id_init_sample}"

    cfg_file, additional_xml = prepare_sumo_files(method=method,annotation=annotation,rs=rs)

    raw_samples=min(100*x_dim,5000)
    batch_size=1
    num_restarts_opti=5
    device=(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    raasp_sampler=RAASP_sampler(x_dim=x_dim,seed=rs)

    sumo_simulator=SumoSimulation(num_replications=num_replications,
                                    cfg_file=cfg_file,
                                    output_dir=output_dir,
                                    tl_ids=tl_ids,
                                    )

    objective_function=ErrorFunction(
        initial_tl_df=initial_tl_df,
        output_dir=output_dir,
        sumo_simulator=sumo_simulator,
        df_gt=df_gt,
        additional_xml=additional_xml,
        tl_ids=tl_ids
    )

    
    BO_instance=BOLoop(niter=niter,
                    num_restart=num_restart,
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
                    RAASP_sampler=raasp_sampler)
    BO_instance.run_loop(version="MSR",initial_length_scale_value=initial_length_scale_value*np.sqrt(x_dim),rs=rs,annotation=annotation)

def run_turbo(method,id_init_sample,initial_length_scale_value,rs,scaled=False):
    set_seed(rs)

    n_trust_regions=1
    x_dim=98
    num_restarts=3
    niter=30
    tr_candidates_generator=TRCandidatesGeneration(x_dim=x_dim,seed=rs)


    output_dir="turbo_output/"
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


    df_gt=pd.read_csv("./gt_data/gt_InTAS.travel_times.csv")
    net=sumolib.net.readNet("Ingolstadt Network/ingolstadt.net.xml")
    tl_ids=[tl.getID() for tl in net.getTrafficLights()]
    num_replications=3

    if scaled:
        annotation=f"ls_{initial_length_scale_value}scaled_is{id_init_sample}"
        initial_length_scale_value=initial_length_scale_value*np.sqrt(x_dim)
    else:
        annotation=f"ls_{initial_length_scale_value}_is{id_init_sample}"

    cfg_file, additional_xml = prepare_sumo_files(method=method,annotation=annotation,rs=rs)

    batch_size=1

    sumo_simulator=SumoSimulation(num_replications=num_replications,
                                    cfg_file=cfg_file,
                                    output_dir=output_dir,
                                    tl_ids=tl_ids,
                                    )

    objective_function=ErrorFunction(
        initial_tl_df=initial_tl_df,
        output_dir=output_dir,
        sumo_simulator=sumo_simulator,
        df_gt=df_gt,
        additional_xml=additional_xml,
        tl_ids=tl_ids
    )

    trust_region_initial_df_list=torch.load(f"./initial_samples/init_df_{id_init_sample}.pt",weights_only=False)
    

    trust_region_initial_y_list=torch.load(f"./initial_samples/init_Y_{id_init_sample}.pt",weights_only=False)
    

    trust_region_initial_x_norm_list=torch.load(f"./initial_samples/init_X_norm_{id_init_sample}.pt",weights_only=False)
    

    trust_region_initial_df_list=[trust_region_initial_df_list]
    trust_region_initial_y_list=[trust_region_initial_y_list]
    trust_region_initial_x_norm_list=[trust_region_initial_x_norm_list]

    torbo_bo_launcher=TurboLoop(
        n_trust_regions=n_trust_regions,
        dim=x_dim,
        num_restarts=num_restarts,
        tr_candidates_generator=tr_candidates_generator,
        objective_function=objective_function,
        trust_region_initial_x_norm_list=trust_region_initial_x_norm_list,
        trust_region_initial_y_list=trust_region_initial_y_list,
        trust_region_initial_df_list=trust_region_initial_df_list,
        batch_size=batch_size,
        niter=niter,
        bounds=bounds,
        output_dir=output_dir
    )


    torbo_bo_launcher.turbo_loop(initial_length_scale_value=initial_length_scale_value,rs=rs,annotation=annotation)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        required=False,
        choices=["vanilla","msr", "turbo"]
    )

    parser.add_argument(
            "--initial_lengthscale",
            type=float,
            required=False
    )


    parser.add_argument(
                    "--scaled",
                    action="store_true",
                    default=False
    )

    parser.add_argument(
        "--experiment_id",
        type=int,
        required=True
    )

    args = parser.parse_args()

    id_init_samples = [0, 1, 2]
    restarts = [0, 1, 2]

    experiments = list(product(id_init_samples, restarts))

    if args.experiment_id >= len(experiments):
        parser.error("Invalid experiment_id")

    id_init_sample, rs = experiments[args.experiment_id]

    print(f"Experiment ID: {args.experiment_id}")
    print(f"Initial sample: {id_init_sample}")
    print(f"Restart: {rs}")


    if args.method=="vanilla":
        run_vanilla(method=args.method,id_init_sample=id_init_sample,initial_length_scale_value=args.initial_lengthscale,rs=rs,scaled=args.scaled)
    elif args.method=="msr":
        run_msr(method=args.method,id_init_sample=id_init_sample,initial_length_scale_value=args.initial_lengthscale,rs=rs,scaled=args.scaled)
    elif args.method=="turbo":
        run_turbo(method=args.method,id_init_sample=id_init_sample,initial_length_scale_value=args.initial_lengthscale,rs=rs,scaled=args.scaled)