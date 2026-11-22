from scipy.integrate import quad,nquad
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from modules.sumo import get_route_gt

class InverseCDF():
    def __init__(self,
                 df_gt,
                 initial_tl_df,
                 bounds
                 ):
        
        self.df_gt=df_gt
        self.df_gt=df_gt.sort_values(by="route_id")

        self.df_gt["route_weight"]=self.df_gt["route_weight"]/self.df_gt["route_weight"].sum()
        
        
        self.cycle_time=90

        self.n_routes=len(self.df_gt["route_id"].unique())
        if self.n_routes!=len(self.df_gt):
            raise ValueError("The ground truth DF should contain one row per route")

        self.bounds=bounds
        self.initial_tl_df=initial_tl_df

        self.tl_ids=np.sort(self.initial_tl_df["tl"].unique())
        self.tl_indices_dict={}
        for tl in self.tl_ids:
            self.tl_indices_dict.setdefault(tl,{})
            df_gt_filtered=self.df_gt.loc[self.df_gt["tl"]==tl].reset_index(drop=True)
            df_gt_filtered=df_gt_filtered.sort_values(by=["route_id"])
            self.tl_indices_dict[tl]["to_vary"]=df_gt_filtered.loc[df_gt_filtered["main_green_phase_to_change"]==1].index
            self.tl_indices_dict[tl]["to_change"]=df_gt_filtered.loc[df_gt_filtered["main_green_phase_to_change"]==2].index
            self.tl_indices_dict[tl]["constant"]=df_gt_filtered.loc[df_gt_filtered["main_green_phase_to_change"]==0].index

    def tl_inverse_cdf(self,u,bounds):
        def objective(g,bounds,u):
            return self.tl_cdf(g,bounds)-u
        g=brentq(objective,args=(bounds,u))

        return g
    

    def tl_cdf(self,g,bounds,tl,z):
        result,_=quad(self.tl_normalized_pdf,a=bounds[0],b=g,args=(tl,z,))
        return result

    def tl_normalized_pdf(self,g,tl,z):
        fA_loss=self.tl_fA_loss(g,tl)
        return (1/z)*np.exp(-fA_loss)


    def compute_Z(self,bounds,tl):
        z,_=quad(self.tl_unnormalized_pdf,a=bounds[0],b=bounds[1],args=(tl,))
        return z

    def tl_unnormalized_pdf(self,g,tl):
        fA_loss=self.tl_fA_loss(g,tl)
        return np.exp(-fA_loss)

    def tl_fA_loss(self,g,tl):

        df_gt_copy=self.df_gt.copy()
        df_gt_copy_filtered=df_gt_copy.loc[df_gt_copy["tl"]==tl].reset_index(drop=True)
        df_gt_copy_filtered=df_gt_copy_filtered.sort_values(by=["route_id"])

        cumulative_const_value=df_gt_copy_filtered["tl_cumulative_constant_duration"].iloc[0]
        gt=np.zeros(len(df_gt_copy_filtered))
        gt[self.tl_indices_dict[tl]["to_vary"]]=g
        gt[self.tl_indices_dict[tl]["to_change"]]=90-g-cumulative_const_value
        gt[self.tl_indices_dict[tl]["constant"]]=df_gt_copy_filtered.iloc[self.tl_indices_dict[tl]["constant"]]["initial_duration"]

        ff_tt_values=df_gt_copy_filtered["route_sum_ff_tt"].values
        length_values=df_gt_copy_filtered["route_sum_length"].values
        ff_speed_values=df_gt_copy_filtered["route_ff_speed"].values

        route_weights_values=df_gt_copy_filtered["route_weight"].values

        true_tt_values=df_gt_copy_filtered["route_average_traveltime"].values

        normalized_critical_gt=(27.79340206/(ff_tt_values-0.10356882)) + 0.10519363

        proport_ff_tt=(gt/self.cycle_time)*ff_tt_values
        inverse_proport_ff_tt=1/proport_ff_tt
        normalized_gt=gt/ff_tt_values
        normalized_critical_gt_diff=normalized_gt-normalized_critical_gt
        critical_ff_speed_length_product=normalized_critical_gt_diff*length_values*ff_speed_values
        square_inverse_proport_ff_tt=np.square(inverse_proport_ff_tt)

        predictors=np.array([inverse_proport_ff_tt,normalized_gt,normalized_critical_gt_diff,ff_tt_values,ff_speed_values,critical_ff_speed_length_product,square_inverse_proport_ff_tt])
        coeffs=np.array([-6.40621988e+00,8.57352254e+01,-7.41941697e+00,1.10365529e+01,1.86797158e-01,4.04908884e-01,-2.79010501e-04,-1.20503437e+01])

        connexion_fA_value=coeffs[0]+np.matmul(coeffs[1:],predictors)
        connexion_tt_fA_values=connexion_fA_value*ff_tt_values

        return np.sum(route_weights_values*np.square(true_tt_values-connexion_tt_fA_values))





"""
    def fA_loss(self,g_vec):
        corridors_df_gt=get_route_gt(g_vec,self.corridors_df,self.initial_tl_df)
        gt=corridors_df_gt["gt"]

    
        proport_ff_tt=(gt/self.cycle_time)*self.ff_tt_values
        inverse_proport_ff_tt=1/proport_ff_tt
        normalized_gt=gt/self.ff_tt_values
        normalized_critical_gt_diff=normalized_gt-self.normalized_critical_gt
        critical_ff_speed_length_product=normalized_critical_gt_diff*self.length_values*self.ff_speed_values
        square_inverse_proport_ff_tt=np.square(inverse_proport_ff_tt)

        predictors=np.array([inverse_proport_ff_tt,normalized_gt,normalized_critical_gt_diff,self.ff_tt_values,self.ff_speed_values,critical_ff_speed_length_product,square_inverse_proport_ff_tt])
        coeffs=np.array([-6.40621988e+00,8.57352254e+01,-7.41941697e+00,1.10365529e+01,1.86797158e-01,4.04908884e-01,-2.79010501e-04,-1.20503437e+01])

        connexion_fA_value=coeffs[0]+np.matmul(coeffs[1:],predictors)
        connexion_tt_fA_values=connexion_fA_value*self.ff_tt_values

        return np.sum(self.route_weights_values*np.square(self.true_tt_values-connexion_tt_fA_values))/self.total_weights

        
"""