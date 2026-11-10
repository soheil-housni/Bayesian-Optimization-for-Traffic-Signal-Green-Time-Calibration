from botorch.optim import optimize_acqf
import torch
from botorch.utils.transforms import unnormalize
from functools import partial
from modules.msr import raasp_ic_generator


def optimize_acqf_and_get_observation(acq_func,
                                        bounds,
                                        batch_size,
                                        device,
                                        dtype,
                                        MSR=False,
                                        raw_samples=None,
                                        raasp_sampler=None,
                                        top_x=None,
                                        i=None,
                                        rs=None,
                                        botlzmann_sampler=None,
                                        num_restarts_opti=5):
        
        """Optimizes the acquisition function, and returns a new candidate."""

        dim = acq_func.model.train_inputs[0].size(dim=1)

        if MSR==True:

            custom_ic_generator = partial(
                                raasp_ic_generator,
                                raasp_sampler=raasp_sampler,
                                top_x=top_x,
                                i=i,
                                rs=rs,
                                botlzmann_sampler=botlzmann_sampler
                            )

            # optimize
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=torch.tensor([[0.0] * dim, [1.0] * dim], device=device, dtype=dtype),
                q=batch_size,
                num_restarts=num_restarts_opti,
                raw_samples=raw_samples,  # used for intialization heuristic
                options={"batch_limit": 5, "maxiter": 200},
                ic_generator=custom_ic_generator
            )

        else:
              candidates, _ = optimize_acqf(
                              acq_function=acq_func,
                              bounds=torch.tensor([[0.0] * dim, [1.0] * dim], device=device, dtype=dtype),
                              q=batch_size,
                              num_restarts=num_restarts_opti,
                              raw_samples=raw_samples,  # used for intialization heuristic
                              options={"batch_limit": 5, "maxiter": 200},
                          )
              

        # observe new values 
        new_x = candidates.detach()

        unnormalized_new_x=unnormalize(new_x, bounds)
        
        return unnormalized_new_x