import torch
from torch.quasirandom import SobolEngine
from botorch.utils.transforms import unnormalize,normalize
from modules.sumo import check_feasibility
import pandas as pd
import numpy as np
from modules.sumo import ErrorFunction

class InitialSamples():
    def __init__(self,
                 dtype,
                 initial_tl_df,
                 objective_function
                 ):
        
        self.dtype=dtype
        self.initial_tl_df=initial_tl_df
        self.objective_function=objective_function

    def get_initial_sample(self,sample_size, x_dim, x_bounds,seed,demand_scaling_factor=None):
        init_X=self.get_x_sample(sample_size=sample_size, x_dim=x_dim, x_bounds=x_bounds,seed=seed)
        init_X_norm, init_Y, init_df=self.get_objective(init_X=init_X,sample_size=sample_size, x_bounds=x_bounds,demand_scaling_factor=demand_scaling_factor)
        return init_X_norm, init_Y, init_df


    def get_objective(self,init_X,sample_size, x_bounds,demand_scaling_factor=None):
            
        init_objective_values = []
        for n1 in range(sample_size):
            print(f"Sample number {n1}")
            epoch_index = f'0{n1}_initial_sample'
            print(epoch_index)
            x = init_X.cpu().detach().numpy()[n1]
            objective_value=self.objective_function.compute_error(g_vec=x,epoch_index=epoch_index,demand_scaling_factor=demand_scaling_factor)
            print("objective_value:")
            print(objective_value)
            init_objective_values.append(objective_value)
            print('*** current green times: ', x)

        
        init_objective_values=np.array(init_objective_values).reshape(sample_size,1)

        columns=[f"gt_{i}" for i in range(98)]
        columns.append("loss_estimate")

        init_df=pd.DataFrame(np.hstack((init_X,init_objective_values)),columns=columns) #
        init_df['iter']=-1
        
        
        train_X = torch.from_numpy(init_df[columns[:-1]].values)
        init_X_norm = normalize(train_X,x_bounds)
        
        init_Y = -torch.from_numpy(init_df[["loss_estimate"]].values)

        return init_X_norm, init_Y, init_df 



    def get_x_sample(self,sample_size, x_dim, x_bounds,seed):

        init_X=[]
        sobol = SobolEngine(x_dim, scramble=True,seed=seed)
        for n_1 in range(sample_size):
            feasibility_bool = False
            while not feasibility_bool:
                norm_x = sobol.draw(1).to(dtype=self.dtype)[0]
                x = unnormalize(norm_x, x_bounds)
                feasibility_bool = check_feasibility(x.cpu().detach().numpy(),self.initial_tl_df)
                print("feasibility_bool:")
                print(feasibility_bool)
            init_X.append(x)
            # map the normalized into the original parameter space
        init_X=torch.stack(init_X,dim=0)

            # check feasibility of all initial points
        return init_X