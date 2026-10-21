from .turbo_state import TurboState
import torch
from .tr_gp_fit import tr_gp_fit
from botorch.generation.sampling import MaxPosteriorSampling
import numpy as np
import pandas as pd
from botorch.utils.transforms import unnormalize,normalize
from .custom_thompson_sampler import CustomThompsonSampler

class TurboLoop():

    def __init__(self,
                 n_trust_regions,
                 dim,
                 num_restarts,
                 niter,
                 tr_candidates_generator,
                 objective_function,
                 trust_region_initial_x_norm_list,
                 trust_region_initial_y_list,
                 trust_region_initial_df_list,
                 bounds,
                 thompson_r_sample=1,
                 ard=False,
                 batch_size=1):

        self.batch_size=batch_size
        self.n_trust_regions=n_trust_regions
        self.dim=dim
        self.num_restarts=num_restarts
        self.niter=niter
        self.tr_candidates_generator=tr_candidates_generator
        

        self.ard=ard

        self.batch_size=batch_size

        self.thompson_r_sample=thompson_r_sample

        self.objective_function=objective_function

        self.trust_region_initial_x_norm_list=trust_region_initial_x_norm_list
        self.trust_region_initial_y_list=trust_region_initial_y_list
        self.trust_region_initial_df_list=trust_region_initial_df_list

        self.bounds=bounds


    def turbo_loop(self):
        for rs in range(self.num_restarts):
            self.TR_turbo_states=[TurboState(dim=self.dim,
                                                tr_id=n,
                                                batch_size=self.batch_size,
                                                tr_init_x_norm=self.trust_region_initial_x_norm_list[rs][n],
                                                tr_init_y=self.trust_region_initial_y_list[rs][n],
                                                tr_init_df=self.trust_region_initial_df_list[rs][n]) for n in range(self.n_trust_regions)]
            for i in range(self.niter):
                batch_candidates_across_tr=[]
                batch_obj_value_across_tr=[]
                batch_selected_tr=[]

                active_trs=[tr for tr in self.TR_turbo_states if not tr.restart_triggered]
                if not active_trs:
                    print(f"No activated regions available => End of the optimization for restart {rs}")
                    break

                self.TR_gp_models=[]

                for n in range(len(active_trs)):
                    tr_state=active_trs[n]
                    tr_id=tr_state.tr_id
                    tr_x_norm=tr_state.tr_x_norm
                    tr_y=tr_state.tr_y
                    tr_df=tr_state.tr_df

                    tr_y_best,tr_y_best_idx=torch.max(tr_y,dim=0)
                    tr_x_best_norm=tr_x_norm[tr_y_best_idx.item()].clone().detach()

                    tr_state.update_best_value(tr_y_best.item())
                    tr_state.tr_x_best_norm=tr_x_best_norm
                    tr_state.tr_x_best=unnormalize(tr_x_best_norm,self.bounds)

                    gp_model=tr_gp_fit(tr_x_norm,tr_y)

                    if not self.ard:
                        tr_state.tr_ub=torch.clamp(tr_x_best_norm+tr_state.length/2,0.0,1.0)
                        tr_state.tr_lb=torch.clamp(tr_x_best_norm-tr_state.length/2,0.0,1.0)
                    else:
                        weights = gp_model.covar_module.base_kernel.lengthscale.squeeze().detach()
                        weights = weights / weights.mean()
                        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
                        tr_state.tr_lb = torch.clamp(tr_x_best_norm - weights * tr_state.length / 2.0, 0.0, 1.0)
                        tr_state.tr_ub = torch.clamp(tr_x_best_norm + weights * tr_state.length / 2.0, 0.0, 1.0)

                    self.TR_gp_models.append(gp_model)

                for b in range(self.batch_size):
                    batch_selection_candidates=[]
                    batch_f_samples=[]
                    batch_tr_id=[]
                    for n in range(len(active_trs)):
                        tr_state=active_trs[n]
                        tr_id=tr_state.tr_id
                        tr_x_norm=tr_state.tr_x_norm
                        tr_y=tr_state.tr_y
                        tr_df=tr_state.tr_df

                        tr_y_best,tr_y_best_idx=torch.max(tr_y,dim=0)
                        tr_x_best_norm=tr_x_norm[tr_y_best_idx.item()].clone().detach()

                        tr_state.update_best_value(tr_y_best.item())
                        tr_state.tr_x_best_norm=tr_x_best_norm
                        tr_state.tr_x_best=unnormalize(tr_x_best_norm,self.bounds)

                        gp_model=tr_gp_fit(tr_x_norm,tr_y)

                        if not self.ard:
                            tr_state.tr_ub=torch.clamp(tr_x_best_norm+tr_state.length/2,0.0,1.0)
                            tr_state.tr_lb=torch.clamp(tr_x_best_norm-tr_state.length/2,0.0,1.0)
                        else:
                            weights = gp_model.covar_module.base_kernel.lengthscale.squeeze().detach()
                            weights = weights / weights.mean()
                            weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
                            tr_state.tr_lb = torch.clamp(tr_x_best_norm - weights * tr_state.length / 2.0, 0.0, 1.0)
                            tr_state.tr_ub = torch.clamp(tr_x_best_norm + weights * tr_state.length / 2.0, 0.0, 1.0)

                        tr_candidates_set=self.tr_candidates_generator.RAASP_sampling(
                            tr_state.tr_x_best_norm,
                            tr_state.tr_lb,
                            tr_state.tr_ub
                        )

                        tr_custom_thompson_sampler=CustomThompsonSampler(gp_model)
                        with torch.no_grad():
                            best_tr_candidate,f_sample_value=tr_custom_thompson_sampler(tr_candidates_set,num_samples=1)
                            best_tr_candidate=best_tr_candidate.squeeze(0).squeeze(0)
                            batch_selection_candidates.append(best_tr_candidate)
                            batch_f_samples.append(f_sample_value)
                            batch_tr_id.append(tr_id)

                    batch_selection_candidates=torch.stack(batch_selection_candidates,dim=0)

                    selected_candidate_idx=np.argmax(batch_f_samples)
                    selected_candidate=batch_selection_candidates[selected_candidate_idx]
                    selected_tr=batch_tr_id[selected_candidate_idx]

                    batch_candidates_across_tr.append(selected_candidate)
                    batch_selected_tr.append(selected_tr)

                    unnorm_selected_candidate=unnormalize(selected_candidate,bounds=self.bounds)
                    y_value=self.objective_function.compute_error(
                        unnorm_selected_candidate.detach().cpu().numpy(),
                        epoch_index=f"turbo_candidate_{b}_tr_{selected_tr}_iter_{i}_rs_{rs}"
                    )
                    batch_obj_value_across_tr.append(-y_value)

                batch_candidates_across_tr=torch.stack(batch_candidates_across_tr)

                for n in range(len(active_trs)):
                    tr_state=active_trs[n]
                    tr_id=tr_state.tr_id
                    tr_candidates=[]
                    tr_obj_values=[]
                    for b in range(len(batch_candidates_across_tr)):
                        candidate=batch_candidates_across_tr[b]
                        obj_value=batch_obj_value_across_tr[b]
                        selected_tr=batch_selected_tr[b]

                        if selected_tr==tr_id:
                            tr_candidates.append(candidate)
                            tr_obj_values.append(obj_value)

                    if tr_candidates:
                        tr_state.update_state(torch.tensor(tr_obj_values))
                        tr_candidates=torch.stack(tr_candidates)
                        tr_obj_values=torch.tensor(tr_obj_values)
                        tr_state.tr_x_norm=torch.cat([tr_state.tr_x_norm,tr_candidates],dim=0)
                        tr_state.tr_y=torch.cat([tr_state.tr_y,tr_obj_values],dim=0)
                        unnorm_x=unnormalize(tr_candidates,bounds=self.bounds)
                        tr_state.tr_df=pd.concat([tr_state.tr_df,
                                                  pd.DataFrame(
                                                      np.hstack((unnorm_x.detach().cpu().numpy(),
                                                                 -tr_obj_values.detach().cpu().numpy().reshape(-1,1),
                                                                 np.full((len(tr_obj_values),1),i))),
                                                                 columns=tr_state.tr_df.columns)],axis=0,ignore_index=True)
            for n in range(len(self.TR_turbo_states)):
                tr_state=self.TR_turbo_states[n]
                tr_id=tr_state.tr_id
                tr_df=tr_state.tr_df
                tr_df.to_csv(f"turbo_tr_{tr_id}_rs_{rs}_training")





                            



                
