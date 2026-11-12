import torch
from torch.quasirandom import SobolEngine
import numpy as np
from botorch.utils.sampling import sample_hypersphere

class RAASP_sampler():
    def __init__(
            self,
            x_dim,
            seed=None
    ):
        self.x_dim=x_dim
        self.sobol_engine=SobolEngine(self.x_dim,seed=seed,scramble=True)
        self.disturbance_prob=min(1,20/x_dim)

    def RAASP_generator(self,
                        top_x,
                        raw_samples):


        m_1=[]
        for i in range(raw_samples//4):
            random_indice=np.random.choice(np.arange(len(top_x)))
            best_x=top_x[random_indice]
            disturbance_vec=pow(10,-3)*torch.randn_like(best_x)
            sampled_x=best_x+disturbance_vec
            invalid=((sampled_x<0) | (sampled_x>1))
            while invalid.any():
                disturbance_vec=pow(10,-3)*torch.randn_like(best_x)
                disturbed_x=best_x+disturbance_vec
                sampled_x=torch.where(invalid,disturbed_x,sampled_x)
                invalid=((sampled_x<0) | (sampled_x>1))

            m_1.append(sampled_x)
        m_1=torch.stack(m_1,dim=0)

        m_2=[]
        for i in range(raw_samples//4):
            random_indice=np.random.choice(np.arange(len(top_x)))
            best_x=top_x[random_indice]

            disturbance_vec=pow(10,-3)*torch.randn_like(best_x)
            disturbed_x=disturbance_vec+best_x

            mask = (
                    torch.rand_like(best_x)
                    < self.disturbance_prob
            )
            
            sampled_x=torch.where(mask,disturbed_x,best_x)
            

            invalid=((sampled_x<0) | (sampled_x>1))
            while invalid.any():
                disturbance_vec=pow(10,-3)*torch.randn_like(best_x)
                disturbed_x=disturbance_vec+best_x
                sampled_x=torch.where(invalid & mask,disturbed_x,sampled_x)
                invalid=((sampled_x<0) | (sampled_x>1))

            m_2.append(sampled_x)



        m_2=torch.stack(m_2,dim=0)

        m_3=self.sobol_engine.draw(raw_samples//2)

        raw_sobol_samples=torch.cat([m_1,m_2,m_3],dim=0)

        return raw_sobol_samples

