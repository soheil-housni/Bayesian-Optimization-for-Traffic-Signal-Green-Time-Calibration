#!/bin/bash

#SBATCH --job-name=test_error
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd "$SLURM_SUBMIT_DIR"

module load python/3.11.5
source ~/envs/thesis/bin/activate


export SUMO_HOME="$HOME/sumo_install/sumo"
export PATH="$SUMO_HOME/bin:$PATH"
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

export LD_PRELOAD="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib/gcc/x86_64-pc-linux-gnu/13/libatomic.so.1"

echo "Python : $(which python)"
python --version

echo "SUMO : $(which sumo)"
sumo --version


python - <<'PY'

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


df_gt=pd.read_csv(f"./gt_data/gt_InTAS.travel_times.csv")
g_vec=np.full(shape=(98,),fill_value=33)


net=sumolib.net.readNet("Ingolstadt Network/ingolstadt.net.xml")
tl_ids=[tl.getID() for tl in net.getTrafficLights()]
additional_xml_gt="./Ingolstadt Network/additional_gt.xml"
initial_tl_df=create_initial_tl_df(additional_xml_gt)
num_replications=3

output_dir_init="initial_output/"


cfg_file="./Ingolstadt Network/InTAS.sumocfg"
output_dir_init="initial_output/"
additional_xml="./Ingolstadt Network/additional.xml"
dtype=torch.double

init_sumo_simulator=SumoSimulation(num_replications=num_replications,
                                   cfg_file=cfg_file,
                                   output_dir=output_dir_init,
                                   tl_ids=tl_ids,
                                   )


error_function=ErrorFunction(
        initial_tl_df=initial_tl_df,
        output_dir=output_dir_init,
        sumo_simulator=init_sumo_simulator,
        df_gt=df_gt,
        additional_xml=additional_xml,
        tl_ids=tl_ids
    )


error_function.compute_error(g_vec=g_vec,epoch_index="test",demand_scaling_factor=3)
PY
