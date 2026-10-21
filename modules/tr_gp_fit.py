import torch
from .gp_initialization import initialize_gp_model
from botorch import fit_gpytorch_mll
from .trust_regions_candidates_generation import TRCandidatesGeneration
from botorch.generation.sampling import MaxPosteriorSampling

def tr_gp_fit(
        tr_x_train,
        tr_y_train,
        ):

        gp_model, gp_mll=initialize_gp_model(tr_x_train,tr_y_train)
        try:
                fit_gpytorch_mll(gp_mll)
                return gp_model
        except:
                print(f'***** CAREFUL ****** kernel was not able to fit GP based , skipping this epoch. Keep in mind for final numerical analysis of results')
                return None
        
        
        
        """
        candidates_set=candidates_generator.RAASP_sampling(tr_x_best,replacement=False)

        thompson_sampling = MaxPosteriorSampling(gp_model)
        with torch.no_grad():
                best_tr_candidate=thompson_sampling(candidates_set,num_samples=batch_size)

        return best_tr_candidate
        """




    

