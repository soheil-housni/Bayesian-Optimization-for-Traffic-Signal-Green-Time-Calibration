from scipy.integrate import quad,nquad
import numpy as np
import pandas as pd

class InverseCDF():
    def __init__(self,
                 df_gt,
                 complete_corridors_connexion_df):
        
        self.df_gt=df_gt
        self.complete_corridors_connexion_df=complete_corridors_connexion_df

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
        self.df_gt["rep_ratio"]=self.df_gt["rep_ratio"]/self.df_gt["rep_ratio"].sum()
        self.df_gt["mse_weight"]=0.75*self.df_gt["connexion_weight"]+0.25*self.df_gt["rep_ratio"]

        self.true_tt_values=self.df_gt["route_connexion_average_traveltime"].values

        self.mse_weights=self.df_gt["mse_weight"].values
        self.ff_tt_values=self.df_gt["route_sum_ff_tt"].values
        self.length_values=self.df_gt["route_sum_length"]
        self.ff_speed_values=self.df_gt["route_ff_speed"]

        self.normalized_critical_gt=(27.79340206/(self.ff_tt_values-0.10356882)) + 0.10519363
        self.cycle_time=90

        self.total_weights=self.mse_weights.sum()

        self.n_connections=len(self.complete_corridors_connexion_df)

        self.bounds=[(1,90)]*self.n_connections


    def compute_CDF(self,
                    connexion_ff_tt,
                    connexion_length,
                    connexion_ff_speed,
                    connexion_normalized_critical_gt,
                    connexion_mse_weight,
                    connexion_true_tt_value,
                    lower_bound,
                    upper_bound):
        
        cdf=quad(
            self.connexion_normalized_pdf,
            a=lower_bound,
            b=upper_bound,
            args=(connexion_ff_tt,
                          connexion_length,
                          connexion_ff_speed,
                          connexion_normalized_critical_gt,
                          connexion_mse_weight,
                          connexion_true_tt_value)
            )

        return cdf

    

    def compute_Z(self,
                  connexion_ff_tt,
                  connexion_length,
                  connexion_ff_speed,
                  connexion_normalized_critical_gt,
                  connexion_mse_weight,
                  connexion_true_tt_value,
                  lower_bound,
                  upper_bound):


            Z = quad(
                    self.connexion_unnormalized_pdf,
                    lower_bound,
                    upper_bound,
                    args=(connexion_ff_tt,
                          connexion_length,
                          connexion_ff_speed,
                          connexion_normalized_critical_gt,
                          connexion_mse_weight,
                          connexion_true_tt_value)
                    )

            return Z


    


    def connexion_normalized_pdf(self,
                                 connexion_gt,
                                 connexion_ff_tt,
                                 connexion_length,
                                 connexion_ff_speed,
                                 connexion_normalized_critical_gt,
                                 connexion_mse_weight,
                                 connexion_true_tt_value,
                                 z):
        
        connexion_fA_loss=self.connexion_fA_loss(connexion_gt,
                                                connexion_ff_tt,
                                                connexion_length,
                                                connexion_ff_speed,
                                                connexion_normalized_critical_gt,
                                                connexion_mse_weight,
                                                connexion_true_tt_value)
        
        return np.exp(- connexion_fA_loss)*1/z
         

    def connexion_unnormalized_pdf(self,
                                   connexion_gt,
                                   connexion_ff_tt,
                                   connexion_length,
                                   connexion_ff_speed,
                                   connexion_normalized_critical_gt,
                                   connexion_mse_weight,
                                   connexion_true_tt_value):
        
        connexion_fA_loss=self.connexion_fA_loss(connexion_gt,
                                                   connexion_ff_tt,
                                                   connexion_length,
                                                   connexion_ff_speed,
                                                   connexion_normalized_critical_gt,
                                                   connexion_mse_weight,
                                                   connexion_true_tt_value)
        return np.exp(- connexion_fA_loss)


    def connexion_fA_loss(self,
                            connexion_gt,
                            connexion_ff_tt,
                            connexion_length,
                            connexion_ff_speed,
                            connexion_normalized_critical_gt,
                            connexion_mse_weight,
                            connexion_true_tt_value):
        
        proport_ff_tt=(connexion_gt/self.cycle_time)*connexion_ff_tt
        inverse_proport_ff_tt=1/proport_ff_tt
        normalized_gt=connexion_gt/connexion_ff_speed
        normalized_critical_gt_diff=normalized_gt-connexion_normalized_critical_gt
        critical_ff_speed_length_product=normalized_critical_gt_diff*connexion_length*connexion_ff_speed
        square_inverse_proport_ff_tt=np.square(inverse_proport_ff_tt)

        predictors=np.array([inverse_proport_ff_tt,normalized_gt,normalized_critical_gt_diff,connexion_ff_tt,connexion_ff_speed,critical_ff_speed_length_product,square_inverse_proport_ff_tt])
        coeffs=np.array([-6.40621988e+00,8.57352254e+01,-7.41941697e+00,1.10365529e+01,1.86797158e-01,4.04908884e-01,-2.79010501e-04,-1.20503437e+01])

        connexion_fA_value=coeffs[0]+np.matmul(coeffs[1:],predictors)
        connexion_tt_fA_values=connexion_fA_value*self.ff_tt_values

        return connexion_mse_weight*np.square(connexion_true_tt_value-connexion_tt_fA_values)

    

    