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


def generate_gt_dataset():
    net=sumolib.net.readNet("Ingolstadt Network/ingolstadt.net.xml")
    tl_ids=[tl.getID() for tl in net.getTrafficLights()]
    num_replications=3
    cfg_file="Ingolstadt Network/InTAS_gt.sumocfg"
    output_dir="gt_data/"

    gt_sumo_simulator=SumoSimulation(num_replications,cfg_file,output_dir,tl_ids)
    epoch_index="gt"
    gt_sumo_simulator.simulation(epoch_index)

def generate_initial_samples():

    df_gt=pd.read_csv("./gt_data/gt_InTAS.travel_times.csv")
    net=sumolib.net.readNet("Ingolstadt Network/ingolstadt.net.xml")
    tl_ids=[tl.getID() for tl in net.getTrafficLights()]

    additional_xml_gt="./Ingolstadt Network/additional_gt.xml"
    initial_tl_df=create_initial_tl_df(additional_xml_gt)
    num_replications=3
    cfg_file="./Ingolstadt Network/InTAS.sumocfg"
    output_dir_init="initial_output/"
    additional_xml="./Ingolstadt Network/additional.xml"
    dtype=torch.double

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
    n_init_samples=1

    init_X_norm_list=[]
    init_Y_list=[]
    init_df_list=[]

    for i in range(n_init_samples):
        init_X_norm, init_Y, init_df=initial_sampler.get_initial_sample(sample_size,x_dim,x_bounds)
        init_X_norm_list.append(init_X_norm)
        init_Y_list.append(init_Y)
        init_df_list.append(init_df)

    torch.save(init_X_norm_list,"./initial_samples/init_X_norm.pt")
    torch.save(init_Y_list,"./initial_samples/init_Y.pt")
    torch.save(init_df_list,"./initial_samples/init_df.pt")


def run_vanilla_baseline():
    niter=50
    num_restarts=3
    init_df_list=torch.load("./initial_samples/init_df.pt",weights_only=False)
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
    cfg_file="./Ingolstadt Network/InTAS.sumocfg"
    additional_xml="./Ingolstadt Network/additional.xml"

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

    n_init_samples=1
    for init_sample in range(n_init_samples):

        init_df=init_df_list[init_sample]
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

        BO_instance.run_loop(version="VANILLA",initial_length_scale_value=0.1,annotation=f"ls_0.1_is{init_sample}")


def run_vanilla_scaled():
    niter=50
    num_restarts=3
    init_df_list=torch.load("./initial_samples/init_df.pt",weights_only=False)
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
    cfg_file="./Ingolstadt Network/InTAS.sumocfg"
    additional_xml="./Ingolstadt Network/additional.xml"

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

    n_init_samples=1
    for init_sample in range(n_init_samples):

        init_df=init_df_list[init_sample]
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

        BO_instance.run_loop(version="VANILLA",initial_length_scale_value=0.1*np.sqrt(x_dim),annotation=f"ls_0.1scaled_is{init_sample}")


def run_vanilla_default():
    niter=50
    num_restarts=3
    init_df_list=torch.load("./initial_samples/init_df.pt",weights_only=False)
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
    cfg_file="./Ingolstadt Network/InTAS.sumocfg"
    additional_xml="./Ingolstadt Network/additional.xml"

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

    n_init_samples=1
    for init_sample in range(n_init_samples):
        init_df=init_df_list[init_sample]

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

        BO_instance.run_loop(version="VANILLA",initial_length_scale_value=0.6931,annotation=f"ls_0.6931_is{init_sample}")

def run_msr():
    niter=50
    num_restart=3
    init_df_list=torch.load("./initial_samples/init_df.pt",weights_only=False)
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
    cfg_file="./Ingolstadt Network/InTAS.sumocfg"
    additional_xml="./Ingolstadt Network/additional.xml"

    raw_samples=min(100*x_dim,5000)
    batch_size=1
    num_restarts_opti=5
    device=(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    raasp_sampler=RAASP_sampler(x_dim)

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

    n_init_samples=1
    for init_sample in range(n_init_samples):
        init_df=init_df_list[init_sample]
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
        
        BO_instance.run_loop(version="MSR",initial_length_scale_value=0.1*np.sqrt(x_dim),annotation=f"ls_0.1scaled_is{init_sample}",demand_scaling_factor=3)

def run_turbo_baseline():
    n_trust_regions=1
    x_dim=98
    num_restarts=3
    niter=50
    tr_candidates_generator=TRCandidatesGeneration(x_dim)


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
    cfg_file="./Ingolstadt Network/InTAS.sumocfg"
    additional_xml="./Ingolstadt Network/additional.xml"

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

    trust_region_initial_df_list=torch.load("./initial_samples/init_df.pt",weights_only=False)
    

    trust_region_initial_y_list=torch.load(f"./initial_samples/init_Y.pt",weights_only=False)
    

    trust_region_initial_x_norm_list=torch.load("./initial_samples/init_X_norm.pt",weights_only=False)
    
    n_init_samples=1
    for init_sample in range(n_init_samples):

        trust_region_initial_df_list=[trust_region_initial_df_list[init_sample]]
        trust_region_initial_y_list=[trust_region_initial_y_list[init_sample]]
        trust_region_initial_x_norm_list=[trust_region_initial_x_norm_list[init_sample]]

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

        torbo_bo_launcher.turbo_loop(initial_length_scale_value=0.1,annotation=f"ls_0.1_is{init_sample}")


def run_turbo_default():
    n_trust_regions=1
    x_dim=98
    num_restarts=3
    niter=50
    tr_candidates_generator=TRCandidatesGeneration(x_dim)


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
    cfg_file="./Ingolstadt Network/InTAS.sumocfg"
    additional_xml="./Ingolstadt Network/additional.xml"

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

    trust_region_initial_df_list=torch.load("./initial_samples/init_df.pt",weights_only=False)
    

    trust_region_initial_y_list=torch.load(f"./initial_samples/init_Y.pt",weights_only=False)
    

    trust_region_initial_x_norm_list=torch.load("./initial_samples/init_X_norm.pt",weights_only=False)
    
    n_init_samples=1
    for init_sample in range(n_init_samples):

        trust_region_initial_df_list=[trust_region_initial_df_list[init_sample]]
        trust_region_initial_y_list=[trust_region_initial_y_list[init_sample]]
        trust_region_initial_x_norm_list=[trust_region_initial_x_norm_list[init_sample]]

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

        torbo_bo_launcher.turbo_loop(initial_length_scale_value=0.6931,annotation=f"ls_0.6931_is{init_sample}")


def run_turbo_scaled():
    n_trust_regions=1
    x_dim=98
    num_restarts=3
    niter=50
    tr_candidates_generator=TRCandidatesGeneration(x_dim)


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
    cfg_file="./Ingolstadt Network/InTAS.sumocfg"
    additional_xml="./Ingolstadt Network/additional.xml"

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

    trust_region_initial_df_list=torch.load("./initial_samples/init_df.pt",weights_only=False)
    

    trust_region_initial_y_list=torch.load(f"./initial_samples/init_Y.pt",weights_only=False)
    

    trust_region_initial_x_norm_list=torch.load("./initial_samples/init_X_norm.pt",weights_only=False)
    
    n_init_samples=1
    for init_sample in range(n_init_samples):

        trust_region_initial_df_list=[trust_region_initial_df_list[init_sample]]
        trust_region_initial_y_list=[trust_region_initial_y_list[init_sample]]
        trust_region_initial_x_norm_list=[trust_region_initial_x_norm_list[init_sample]]

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

        torbo_bo_launcher.turbo_loop(initial_length_scale_value=0.1*np.sqrt(98),annotation=f"ls_0.1scaled_is{init_sample}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["generate_gt_dataset","generate_initial_samples","vanilla_baseline", "vanilla_scaled", "vanilla_default","msr", "turbo_baseline","turbo_scaled","turbo_default"]
    )

    args = parser.parse_args()

    if args.method=="generate_gt_dataset":
        generate_gt_dataset()
    elif args.method=="generate_initial_samples":
        generate_initial_samples()
    elif args.method=="vanilla_baseline":
        run_vanilla_baseline()
    elif args.method=="vanilla_default":
        run_vanilla_default()
    elif args.method=="vanilla_scaled":
            run_vanilla_scaled()
    elif args.method=="msr":
        run_msr()
    elif args.method=="turbo_baseline":
        run_turbo_baseline()
    elif args.method=="turbo_scaled":
        run_turbo_scaled()
    elif args.method=="turbo_default":
        run_turbo_default()