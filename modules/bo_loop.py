from botorch.acquisition import qLogExpectedImprovement
from torch.quasirandom import SobolEngine
import numpy as np
import pandas as pd
import torch
from botorch.sampling.stochastic_samplers import StochasticSampler
from botorch.utils.transforms import unnormalize,normalize
from .gp_initialization import initialize_gp_model
from botorch import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement
from .feasibility_check import check_feasibility
from .objective_function import ErrorFunction
from .acquisition_function_optimization import optimize_acqf_and_get_observation
from .ic_generator import raasp_ic_generator

class BOLoop():
    def __init__(self,
                 niter,
                 num_restart,
                 init_df_list,
                 output_dir,
                 bounds,
                 objective_function,
                 dtype,
                 x_dim,
                 batch_size,
                 raw_samples,
                 device,
                 initial_tl_df,
                 num_restarts_opti=5,
                 RAASP_sampler=None,
                 ):
        
        self.niter=niter
        self.num_restart=num_restart
        self.init_df_list=init_df_list
        self.output_dir=output_dir
        self.bounds=bounds
        self.objective_function=objective_function
        self.dtype=dtype
        self.x_dim=x_dim
        self.batch_size=batch_size
        self.raw_samples=raw_samples
        self.num_restarts_opti=num_restarts_opti
        self.device=device
        self.initial_tl_df=initial_tl_df
        self.RAASP_sampler=RAASP_sampler
    

    def run_loop(self,
                 version,
                 initial_length_scale_value=0.6931
                 ):

        for rs in range(self.num_restart):
            best_value = []
            for i in range(self.niter):
                df = self.init_df_list[rs] if i == 0 else df
                n_obs = df.shape[0]
                train_X= torch.from_numpy(df[[f"gt_{i}" for i in range(self.x_dim)]].values)
                train_X_norm = normalize(train_X,self.bounds)
                train_X_norm=train_X_norm.to(self.dtype)
                train_Y = -torch.from_numpy(df[["loss_estimate"]].values)
                train_Y=train_Y.to(self.dtype)

                best_y,best_y_index = torch.max(train_Y,dim=0)
                best_y=best_y.item()
                best_y_index=best_y_index.item()

                x_best=train_X_norm[best_y_index]

                torch.save(x_best,f"./best_x/x_best_iter_{i}_restart_{rs}.pt")

                best_value.append(best_y)

                print(f"##### best_value={best_y} #####")
                print(f"Shape of train_X_norm is {train_X_norm.shape}")
                gp_model, gp_mll = initialize_gp_model(train_X_norm,train_Y,initial_length_scale_value=0.6931)
                try:
                    fit_gpytorch_mll(gp_mll)
                except:
                    print(f'***** CAREFUL ****** kernel was not able to fit GP based at epoch {i}, skipping this epoch. Keep in mind for final numerical analysis of results')
                    continue

                epoch_index=f"BO_iter_{i}_restart_{rs}"

                if version=="VANILLA":

                    sampler = StochasticSampler(sample_shape=torch.Size([128]))
                    EI = LogExpectedImprovement(gp_model, best_f=best_y, maximize=True)
                    new_x = optimize_acqf_and_get_observation(acq_func=EI,
                                                            bounds=self.bounds,
                                                            batch_size=self.batch_size,
                                                            device=self.device,
                                                            dtype=self.dtype,
                                                            raw_samples=self.raw_samples,
                                                            num_restarts_opti=self.num_restarts_opti
                                                            )

                elif version=="MSR":

                    sampler = StochasticSampler(sample_shape=torch.Size([128]))
                    EI = LogExpectedImprovement(gp_model, best_f=best_y, maximize=True)

                    new_x = optimize_acqf_and_get_observation(acq_func=EI,
                                                            bounds=self.bounds,
                                                            batch_size=self.batch_size,
                                                            device=self.device,
                                                            dtype=self.dtype,
                                                            num_restarts_opti=self.num_restarts_opti,
                                                            raasp_sampler=self.RAASP_sampler,
                                                            x_best=x_best,
                                                            raw_samples=self.raw_samples,
                                                            i=i,
                                                            rs=rs,
                                                            MSR=True
                                                            )

            
                print("------------------------------")
                print(new_x)
                print("------------------------------")


                if not check_feasibility(new_x.cpu().detach().numpy().reshape(-1),initial_tl_df=self.initial_tl_df):
                    print('*** BO optimal point is not feasible, replacing w/ a randomly sampled point')
                    is_feasible = False
                    while not is_feasible:
                        sobol = SobolEngine(self.x_dim, scramble=True)
                        x1 = sobol.draw(1).to(dtype=self.dtype)
                        new_x = unnormalize(x1, self.bounds)
                        if check_feasibility(new_x.cpu().detach().numpy().reshape(-1)):
                            is_feasible = True                        
                
                train_new_x=new_x.cpu().detach().numpy().reshape(-1)

                new_obj = self.objective_function.compute_error(train_new_x,epoch_index=epoch_index)

                new_row=pd.DataFrame(np.append(train_new_x,[new_obj,i+1]).reshape(1,-1),columns=[f"gt_{i}" for i in range(self.x_dim)]+["loss_estimate","iter"])
            
                df = pd.concat([df, new_row], axis=0, ignore_index=True)
            
            
            # print outputs
            # if kernel_name == 'custom_Matern': 
            #     df.to_csv(f'{OUTPUT_DIR}df_custom_restart{rs}.csv', index=None)    
            # if kernel_name == 'Matern_default_v1':
            #     df.to_csv(f'{OUTPUT_DIR}df_default_restart{rs}.csv', index=None)    
            # if kernel_name == 'custom_Matern_v2':
            df.to_csv(f'{self.output_dir}BO_df_{version}_restart{rs}.csv', index=None)    
        

            






