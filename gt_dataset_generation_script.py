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
from modules.sumo import create_gt_dataset

def generate_gt_dataset():
    net=sumolib.net.readNet("Ingolstadt Network/ingolstadt.net.xml")
    tl_ids=[tl.getID() for tl in net.getTrafficLights()]
    num_replications=10
    cfg_file="Ingolstadt Network/InTAS_gt.sumocfg"
    output_dir="gt_data/"

    gt_sumo_simulator=SumoSimulation(num_replications,cfg_file,output_dir,tl_ids)
    epoch_index="gt"
    create_gt_dataset(output_dir=output_dir,epoch_index=epoch_index,sumo_simulator=gt_sumo_simulator,num_replications=num_replications)



if __name__=="__main__":
    
    generate_gt_dataset()