import torch
from torch.quasirandom import SobolEngine
import numpy as np
from botorch.utils.sampling import sample_hypersphere

class RAASP_sampler():
    def __init__(
            self,
            x_dim
    ):
        self.x_dim=x_dim
        self.sobol_engine=SobolEngine(self.x_dim,scramble=True)
        self.disturbance_prob=min(1,20/x_dim)

    def RAASP_generator(self,
                        x_best,
                        raw_samples):
        
        raw_sobol_samples=self.sobol_engine.draw(raw_samples)

        x_best=x_best.unsqueeze(0)

        mask = (
            torch.rand_like(raw_sobol_samples)
            > self.disturbance_prob
        )

        raw_sobol_samples=torch.where(mask,x_best,raw_sobol_samples)

        return raw_sobol_samples

