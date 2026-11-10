from functools import partial
import torch
from botorch.optim import optimize_acqf
from botorch.optim.initializers import initialize_q_batch


def raasp_ic_generator(
    acq_function,
    bounds,
    q,
    num_restarts,
    raw_samples,
    raasp_sampler=None,
    top_x=None,
    i=None,
    rs=None,
    botlzmann_sampler=None,
    **kwargs,
):

    X_raw = raasp_sampler.RAASP_generator(top_x,raw_samples)

    #X_raw = X_raw.unsqueeze(-2)

    X_raw = X_raw.unsqueeze(0)
    print(X_raw.shape)
    batch_initial_conditions=botlzmann_sampler(X=X_raw,num_samples=num_restarts)
    batch_initial_conditions=torch.transpose(batch_initial_conditions,0,1)
    print(batch_initial_conditions.shape)



    """
    with torch.no_grad():
        acq_values = acq_function(X_raw)

    batch_initial_conditions,_ = initialize_q_batch(
        X=X_raw,
        acq_vals=acq_values,
        n=num_restarts,
    )
    """
    
    return batch_initial_conditions