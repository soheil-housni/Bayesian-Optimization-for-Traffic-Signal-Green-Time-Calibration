from scipy.integrate import quad,nquad
import numpy as np
import pandas as pd
from scipy.optimize import brentq

class InverseCDF():
    def __init__(self,
                 df_gt
                 ):
        
        self.df_gt=df_gt
        self.df_gt=df_gt.sort_values(by="route_id")

        self.df_gt["route_weight"]=self.df_gt["route_weight"]/self.df_gt["route_weight"].sum()
        
        self.route_weights_values=self.df_gt["route_weight"].values

        self.true_tt_values=self.df_gt["route_average_traveltime"].values

        self.ff_tt_values=self.df_gt["route_sum_ff_tt"].values
        self.length_values=self.df_gt["route_sum_length"]
        self.ff_speed_values=self.df_gt["route_ff_speed"]

        self.normalized_critical_gt=(27.79340206/(self.ff_tt_values-0.10356882)) + 0.10519363
        self.cycle_time=90

        self.total_weights=self.connexion_weights_values.sum()

        self.n_routes=len(self.df_gt["route_id"].unique())
        if self.n_routes!=len(self.df_gt):
            raise ValueError("The ground truth DF should contain one row per route")

        self.bounds=[(1,90)]*self.n_routes





    def fA_loss(self,
                connexion_gt,
                connexion_ff_tt,
                connexion_length,
                connexion_ff_speed,
                connexion_normalized_critical_gt,
                connexion_weight,
                connexion_true_tt_value):
    
    proport_ff_tt=(connexion_gt/self.cycle_time)*connexion_ff_tt
    inverse_proport_ff_tt=1/proport_ff_tt
    normalized_gt=connexion_gt/connexion_ff_tt
    normalized_critical_gt_diff=normalized_gt-connexion_normalized_critical_gt
    critical_ff_speed_length_product=normalized_critical_gt_diff*connexion_length*connexion_ff_speed
    square_inverse_proport_ff_tt=np.square(inverse_proport_ff_tt)

    predictors=np.array([inverse_proport_ff_tt,normalized_gt,normalized_critical_gt_diff,connexion_ff_tt,connexion_ff_speed,critical_ff_speed_length_product,square_inverse_proport_ff_tt])
    coeffs=np.array([-6.40621988e+00,8.57352254e+01,-7.41941697e+00,1.10365529e+01,1.86797158e-01,4.04908884e-01,-2.79010501e-04,-1.20503437e+01])

    connexion_fA_value=coeffs[0]+np.matmul(coeffs[1:],predictors)
    connexion_tt_fA_values=connexion_fA_value*connexion_ff_tt

    return connexion_weight*np.square(connexion_true_tt_value-connexion_tt_fA_values)