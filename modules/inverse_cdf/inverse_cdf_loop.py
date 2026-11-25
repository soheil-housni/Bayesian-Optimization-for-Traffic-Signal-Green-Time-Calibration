import random
import numpy as np
import torch
import pandas as pd
from torch.quasirandom import SobolEngine
from botorch.utils.transforms import unnormalize,normalize
from modules.sumo import check_feasibility

class InverseCDFLoop():
    def __init__(self,
                 cdf_inversor,
                 objective_function,
                 niter,
                 num_replications,
                 initial_tl_df,
                 x_bounds,
                 output_dir,
                 x_dim=98,
                 
                 ):

        self.cdf_inversor=cdf_inversor
        self.objective_function=objective_function
        self.niter=niter
        self.num_replications=num_replications
        self.x_dim=x_dim
        self.initial_tl_df=initial_tl_df
        self.x_bounds=x_bounds
        self.output_dir=output_dir

    def run_loop(self,
                 rs):
        
        self.set_seed(rs)

        df_result=None
        for i in range(self.niter):
            print(f"---ITER # {i}---")
            g_vec=self.cdf_inversor.sample_g_vec()

            if not check_feasibility(g_vec,initial_tl_df=self.initial_tl_df):
                print("Solution drawn not feasible")
                is_feasible = False
                while not is_feasible:
                    g_vec=self.cdf_inversor.sample_g_vec()
                    if check_feasibility(g_vec=g_vec,initial_tl_df=self.initial_tl_df):
                        is_feasible = True
            

            epoch_index=f"Inverse_CDF_sample_{i}_restart{rs}"
            num_replications=self.num_replications
            objective_value=self.objective_function.compute_error(g_vec=g_vec,
                                                                  epoch_index=epoch_index,
                                                                  num_replications=num_replications)

            g_vec=g_vec.reshape(-1)
            row_values=np.append(g_vec,[objective_value,i]).reshape(1,-1)
            row=pd.DataFrame(data=row_values,columns=[f"gt_{g}" for g in range(self.x_dim)]+["loss_estimate","iter"])

            if df_result is None:
                df_result=row
            else:
                df_result=pd.concat([df_result,row],axis=0)

        
        df_result.to_csv(f"{self.output_dir}/inverse_cdf_df_rs{rs}.csv")
            





    def set_seed(self,
                 seed):
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
