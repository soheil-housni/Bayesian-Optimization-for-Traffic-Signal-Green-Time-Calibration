from scipy.integrate import quad,nquad
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from modules.sumo import get_route_gt
from scipy.optimize import minimize_scalar

class InverseCDF():
    def __init__(self,
                 df_gt,
                 initial_tl_df,
                 bounds,
                 route_green_time_accessor,
                 x_dim
                 ):
        
        self.df_gt=df_gt
        self.initial_tl_df=initial_tl_df

        self.route_green_time_accessor=route_green_time_accessor
        self.df_gt=route_green_time_accessor.get_route_green_time(self.df_gt)
        self.df_gt=pd.merge(self.df_gt,self.initial_tl_df[["tl","state","tl_cumulative_constant_duration"]].rename(columns={"state":"main_green_phase"}),on=["tl","main_green_phase"],how="left")
        self.df_gt=self.df_gt.drop_duplicates(subset=["route_str_str"])

        self.df_gt=self.df_gt.sort_values(by="route_id")

        self.df_gt["route_weight"]=self.df_gt["route_weight"]/self.df_gt["route_weight"].sum()

        print(self.df_gt.columns)
        
        
        self.cycle_time=90

        self.n_routes=len(self.df_gt["route_id"].unique())
        if self.n_routes!=len(self.df_gt):
            raise ValueError("The ground truth DF should contain one row per route")

        self.bounds=bounds
        
        self.tl_ids=np.sort(self.initial_tl_df["tl"].unique())
        self.tl_indices_dict={}
        for tl in self.tl_ids:
            self.tl_indices_dict.setdefault(tl,{})
            df_gt_filtered=self.df_gt.loc[self.df_gt["tl"]==tl].reset_index(drop=True)
            df_gt_filtered=df_gt_filtered.sort_values(by=["route_id"])
            self.tl_indices_dict[tl]["to_vary"]=list(df_gt_filtered.loc[df_gt_filtered["main_green_phase_to_change"]==1].index)
            self.tl_indices_dict[tl]["to_change"]=list(df_gt_filtered.loc[df_gt_filtered["main_green_phase_to_change"]==2].index)
            self.tl_indices_dict[tl]["constant"]=list(df_gt_filtered.loc[df_gt_filtered["main_green_phase_to_change"]==0].index)

        self.x_dim=x_dim


        self.min_loss = {}

        for i, tl in enumerate(self.tl_ids):
            bounds_tl = self.bounds[:, i]
            self.min_loss[tl] = self.compute_min_loss(bounds_tl, tl)


    def sample_g_vec(self):
        g_vec=[]
        for i in range(len(self.tl_ids)):
            tl=self.tl_ids[i]
            if tl not in self.df_gt["tl"].unique():
                g_vec.append(-1)
                continue
            bounds=self.bounds[:,i]
            z=self.compute_Z(bounds,tl)
            u=np.random.rand()
            g=self.tl_inverse_cdf(u=u,bounds=bounds,tl=tl,z=z)
            g_vec.append(g)

        return g_vec

    

    def tl_inverse_cdf(self,u,bounds,tl,z):
        def objective(g,bounds,u,tl,z):
            return self.tl_cdf(g,bounds,tl,z)-u
        g=brentq(objective,args=(bounds,u,tl,z),a=bounds[0],b=bounds[1])

        return g

    def tl_cdf(self,g,bounds,tl,z):
        result,_=quad(self.tl_normalized_pdf,a=bounds[0],b=g,args=(tl,z,))
        return result

    def tl_normalized_pdf(self,g,tl,z):
        fA_loss=self.tl_fA_loss(g,tl)
        return (1/z)*np.exp(-(fA_loss-self.min_loss[tl]))


    def compute_Z(self,bounds,tl):
        z,_=quad(self.tl_unnormalized_pdf,a=bounds[0],b=bounds[1],args=(tl,))
        return z

    def tl_unnormalized_pdf(self,g,tl):
        fA_loss=self.tl_fA_loss(g,tl)
        return np.exp(-(fA_loss-self.min_loss[tl]))

    def tl_fA_loss(self,g,tl):

        df_gt_copy=self.df_gt.copy()
        df_gt_copy_filtered=df_gt_copy.loc[df_gt_copy["tl"]==tl].reset_index(drop=True)
        if not df_gt_copy_filtered:
            return None
        df_gt_copy_filtered=df_gt_copy_filtered.sort_values(by=["route_id"])

        print("df_gt_copy_filtered columns :")
        print(df_gt_copy_filtered.columns)

        cumulative_const_value=df_gt_copy_filtered["tl_cumulative_constant_duration"].iloc[0]
        gt=np.zeros(len(df_gt_copy_filtered))
        gt[self.tl_indices_dict[tl]["to_vary"]]=g
        gt[self.tl_indices_dict[tl]["to_change"]]=90-g-cumulative_const_value
        gt[self.tl_indices_dict[tl]["constant"]]=df_gt_copy_filtered.iloc[self.tl_indices_dict[tl]["constant"]]["main_green_phase_initial_duration"]

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


    def compute_min_loss(self, bounds, tl, n_points=1000):

        bounds = np.asarray(bounds, dtype=np.float64)

        g_values = np.linspace(
            float(bounds[0]),
            float(bounds[1]),
            n_points
        )

        loss_values = np.asarray(
            [self.tl_fA_loss(g, tl) for g in g_values],
            dtype=np.float64
        )

        candidates = [np.argmin(loss_values)]

        candidates += list(
            np.where(
                (loss_values[1:-1] < loss_values[:-2]) &
                (loss_values[1:-1] < loss_values[2:])
            )[0] + 1
        )

        min_loss = min(loss_values[0], loss_values[-1])

        for idx in candidates:
            left = g_values[max(0, idx - 1)]
            right = g_values[min(n_points - 1, idx + 1)]

            if left == right:
                continue


            result = minimize_scalar(
                lambda g: self.tl_fA_loss(g, tl),
                bounds=(left, right),
                method="bounded"
            )

            min_loss = min(min_loss, result.fun)

        return min_loss





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