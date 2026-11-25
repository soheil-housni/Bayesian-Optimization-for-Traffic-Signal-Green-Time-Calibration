import numpy as np
import sumolib
import pandas as pd
import torch
from modules.sumo import ExtractRoutesGreenTimes


class UpdatedFAFunctionLinearRegresion():
    def __init__(self,
                 df_gt,
                 initial_tl_df,
                 net_file,
                 dtype=torch.double,
                 cycle_time=90,
                 scaling_loss_factor=1):

        self.cycle_time=cycle_time
        self.scaling_loss_factor=scaling_loss_factor
        self.dtype=dtype

        self.df_gt=df_gt.sort_values(by="route_id")

        self.df_gt["route_weight"]=self.df_gt["route_weight"]/self.df_gt["route_weight"].sum()



        self.initial_tl_df=initial_tl_df
        self.net_file=net_file
        self.routes_green_time_extractor=ExtractRoutesGreenTimes(df_gt=self.df_gt,
                                                        net_file=self.net_file,
                                                        initial_tl_df=self.initial_tl_df)
        




    def fA(self, x_init):
            self.route_traveltimes_true_values=torch.tensor(self.df_gt["route_average_traveltime"].values,dtype=self.dtype).reshape(*([1]*(x_init.ndim-1)),-1)
            self.route_weights=torch.tensor(self.df_gt["route_weight"].values,dtype=self.dtype).reshape(*([1]*(x_init.ndim-1)),-1)
            predictors=self.predictors_generation(x_init)
            predicted_route_traveltime_values=self.extract_route_traveltime(x_init,predictors)
            route_mse=torch.square((predicted_route_traveltime_values/self.scaling_loss_factor)-(self.route_traveltimes_true_values/self.scaling_loss_factor)).sum(dim=-1,keepdim=True)
            return route_mse

    def fA_traveltime(self, x_init):
         predictors=self.predictors_generation(x_init)
         predicted_route_traveltime_values=self.extract_route_traveltime(x_init,predictors)
         return predicted_route_traveltime_values

    def extract_route_traveltime(self,x_init,predictors):
        predicted_route_delay_factor_values=self.linear_regression_model(predictors)
        predicted_route_traveltime_values=predicted_route_delay_factor_values*self.route_ff_tt
        return predicted_route_traveltime_values
        

    def linear_regression_model(self,predictors):

            self.coeffs=(-6.40621988e+00,8.57352254e+01,-7.41941697e+00,1.10365529e+01,1.86797158e-01,4.04908884e-01,-2.79010501e-04,-1.20503437e+01)
            #coeffs=(-8.11543417e+00,2.68213576e+01,-3.15368168e-01,1.43056294e+00,\
                    #1.70163915e-02,8.80479001e-01,-1.50335760e-03,-1.22542459e+00)

            b0,b1,b2,b3,b4,b5,b6,b7=self.coeffs
    
            coeff=torch.tensor([b0,b1,b2,b3,b4,b5,b6,b7],dtype=self.dtype).unsqueeze(1)
    
            result=coeff[0]+torch.matmul(predictors,coeff[1:])
            result[result<0]=1
    
            return result.squeeze(-1)
    
    
    def predictors_generation(self,x_init):
        self.route_ff_tt=torch.tensor(self.df_gt["route_sum_ff_tt"].values,dtype=self.dtype).reshape(*([1]*(x_init.ndim-1)),-1)
        self.route_sum_length=torch.tensor(self.df_gt["route_sum_length"].values,dtype=self.dtype).reshape(*([1]*(x_init.ndim-1)),-1)
        self.route_average_ff_speed=torch.tensor(self.df_gt["route_ff_speed"].values,dtype=self.dtype).reshape(*([1]*(x_init.ndim-1)),-1)


        routes_green_duration_vector=self.routes_green_time_extractor.get_routes_green_times(x_init=x_init)

        proport_ff_tt=(routes_green_duration_vector/self.cycle_time)*self.route_ff_tt
        inverse_proport_ff_tt=1/proport_ff_tt
        normalized_gt=routes_green_duration_vector/self.route_ff_tt
        normalized_critical_gt=(27.79340206/(self.route_ff_tt-0.10356882)) + 0.10519363
        normalized_critical_gt_diff=normalized_gt-normalized_critical_gt
        critical_ff_speed_length_product=normalized_critical_gt_diff*self.route_sum_length*self.route_average_ff_speed
        square_inverse_proport_ff_tt=torch.square(inverse_proport_ff_tt)

        route_ff_tt_expanded=self.route_ff_tt.expand(*routes_green_duration_vector.shape[:-1],-1)
        route_average_ff_speed_expanded=self.route_average_ff_speed.expand(*routes_green_duration_vector.shape[:-1],-1)

        predictors=torch.stack([inverse_proport_ff_tt,normalized_gt,normalized_critical_gt_diff,route_ff_tt_expanded,route_average_ff_speed_expanded,critical_ff_speed_length_product,square_inverse_proport_ff_tt],dim=-1)
        predictors=predictors.to(dtype=self.dtype)

        return predictors

        