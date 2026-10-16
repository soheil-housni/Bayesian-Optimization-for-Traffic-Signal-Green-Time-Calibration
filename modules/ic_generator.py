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
    x_best=None,
    i=None,
    rs=None,
    **kwargs,
):

    X_raw = raasp_sampler.RAASP_generator(x_best,raw_samples)

    X_raw = X_raw.unsqueeze(-2)

    with torch.no_grad():
        acq_values = acq_function(X_raw)

    batch_initial_conditions,_ = initialize_q_batch(
        X=X_raw,
        acq_vals=acq_values,
        n=num_restarts,
    )

    torch.save(batch_initial_conditions,f"./initial_candidates/batch_initial_conditions_iter_{i}_restart_{rs}.pt")

    return batch_initial_conditions