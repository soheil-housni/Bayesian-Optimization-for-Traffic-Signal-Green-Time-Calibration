from scipy.integrate import quad,nquad
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from .green_time_mapping import GreenTimeMapping

class InverseCDF():
    def __init__(self,
                 df_gt,
                 initial_tl_df
                 ):
        
        self.df_gt=df_gt

        self.df_gt=df_gt.sort_values(by="connexion_id")
        
        all_sub_df_gt=None
        for j in range(10):
            sub_df_gt=pd.read_csv(f"./gt_data/complete_corridors_connexions_df_copy_{j}.csv")
            if all_sub_df_gt is None:
                all_sub_df_gt=sub_df_gt
            else:
                all_sub_df_gt=pd.concat([all_sub_df_gt,sub_df_gt],axis=0)

        connexion_id_counts=all_sub_df_gt["connexion_id"].value_counts().reset_index()
        connexion_id_counts["rep_ratio"]=connexion_id_counts["count"]/10

        self.df_gt=pd.merge(self.df_gt,connexion_id_counts,how="left",on="connexion_id")

        self.df_gt["connexion_weight"]=self.df_gt["connexion_weight"]/self.df_gt["connexion_weight"].sum()

        self.connexion_weights_values=self.df_gt["connexion_weight"].values

        self.true_tt_values=self.df_gt["route_connexion_average_traveltime"].values

        self.ff_tt_values=self.df_gt["route_sum_ff_tt"].values
        self.length_values=self.df_gt["route_sum_length"]
        self.ff_speed_values=self.df_gt["route_ff_speed"]

        self.normalized_critical_gt=(27.79340206/(self.ff_tt_values-0.10356882)) + 0.10519363
        self.cycle_time=90

        self.total_weights=self.connexion_weights_values.sum()

        self.n_connections=len(self.df_gt)

        self.bounds=[(1,90)]*self.n_connections

        self.initial_tl_df=initial_tl_df
        self.gt_mapper=GreenTimeMapping(self.df_gt,self.initial_tl_df)

    def connexion_gt_sampling(self):
        all_g=[]

        for i in range(self.n_connections):
            connexion_ff_tt=self.ff_tt_values[i]
            connexion_length=self.length_values[i]
            connexion_ff_speed=self.ff_speed_values[i]
            connexion_normalized_critical_gt=self.normalized_critical_gt[i]
            connexion_weight=self.connexion_weights_values[i]
            connexion_true_tt_value=self.true_tt_values[i]
            lower_bound=self.bounds[i][0]
            upper_bound=self.bounds[i][1]

            z=self.compute_Z(connexion_ff_tt=connexion_ff_tt,
                              connexion_length=connexion_length,
                              connexion_ff_speed=connexion_ff_speed,
                              connexion_normalized_critical_gt=connexion_normalized_critical_gt,
                              connexion_weight=connexion_weight,
                              connexion_true_tt_value=connexion_true_tt_value,
                              lower_bound=lower_bound,
                              upper_bound=upper_bound)

            u=np.random.rand()
            
            g_sample=self.compute_inverse_cdf(u=u,
                                              connexion_ff_tt=connexion_ff_tt,
                                            connexion_length=connexion_length,
                                            connexion_ff_speed=connexion_ff_speed,
                                            connexion_normalized_critical_gt=connexion_normalized_critical_gt,
                                            connexion_weight=connexion_weight,
                                            connexion_true_tt_value=connexion_true_tt_value,
                                            lower_bound=lower_bound,
                                            upper_bound=upper_bound,
                                            z=z)

            all_g.append(g_sample)

        return all_g
            


    def compute_inverse_cdf(self,
                            u,
                            connexion_ff_tt,
                            connexion_length,
                            connexion_ff_speed,
                            connexion_normalized_critical_gt,
                            connexion_weight,
                            connexion_true_tt_value,
                            lower_bound,
                            upper_bound,
                            z):
        def objective(g):
            
            return self.compute_CDF(g,
                                    connexion_ff_tt,
                                    connexion_length,
                                    connexion_ff_speed,
                                    connexion_normalized_critical_gt,
                                    connexion_weight,
                                    connexion_true_tt_value,
                                    lower_bound,
                                    z)-u

        return brentq(objective,a=lower_bound,b=upper_bound)
        


    def compute_CDF(self,
                    g,
                    connexion_ff_tt,
                    connexion_length,
                    connexion_ff_speed,
                    connexion_normalized_critical_gt,
                    connexion_weight,
                    connexion_true_tt_value,
                    lower_bound,
                    z
                    ):
        
        cdf,_=quad(
            self.connexion_normalized_pdf,
            a=lower_bound,
            b=g,
            args=(connexion_ff_tt,
                    connexion_length,
                    connexion_ff_speed,
                    connexion_normalized_critical_gt,
                    connexion_weight,
                    connexion_true_tt_value,
                    z)
            )

        return cdf

    def connexion_normalized_pdf(self,
                                connexion_gt,
                                connexion_ff_tt,
                                connexion_length,
                                connexion_ff_speed,
                                connexion_normalized_critical_gt,
                                connexion_weight,
                                connexion_true_tt_value,
                                z):
        
        connexion_fA_loss=self.connexion_fA_loss(connexion_gt,
                                                connexion_ff_tt,
                                                connexion_length,
                                                connexion_ff_speed,
                                                connexion_normalized_critical_gt,
                                                connexion_weight,
                                                connexion_true_tt_value)
        
        return np.exp(- connexion_fA_loss)*1/z

    

    def compute_Z(self,
                  ):


            z,_ = nquad()
            return z


         

    def connexion_unnormalized_pdf(self,*args):

        g_vec=np.array([*args])
        
        connexion_fA_loss=self.connexion_fA_loss(g_vec)
        return np.exp(- connexion_fA_loss)


    def connexion_fA_loss(self,
                            g_vec,
                            ):

        gt_connexion=self.gt_mapper.compute_connexions_green_times(g_vec)
        
        proport_ff_tt=(gt_connexion/self.cycle_time)*self.ff_tt_values
        inverse_proport_ff_tt=1/proport_ff_tt
        normalized_gt=gt_connexion/self.ff_tt_values
        normalized_critical_gt_diff=normalized_gt-self.normalized_critical_gt
        critical_ff_speed_length_product=normalized_critical_gt_diff*self.length_values*self.ff_speed_values
        square_inverse_proport_ff_tt=np.square(inverse_proport_ff_tt)

        predictors=np.vstack([inverse_proport_ff_tt,normalized_gt,normalized_critical_gt_diff,self.ff_tt_values,self.ff_speed_values,critical_ff_speed_length_product,square_inverse_proport_ff_tt])
        coeffs=np.array([-6.40621988e+00,8.57352254e+01,-7.41941697e+00,1.10365529e+01,1.86797158e-01,4.04908884e-01,-2.79010501e-04,-1.20503437e+01])

        fA_values=coeffs[0]+np.matmul(coeffs[1:],predictors)
        tt_fA_values=fA_values*self.ff_tt_values

        return self.connexion_weights_values*np.square(self.true_tt_values-tt_fA_values).sum()/self.total_weights

    

    