import torch
from torch.quasirandom import SobolEngine

class TRCandidatesGeneration():

    def __init__(self,
                 x_dim,
                 ):
        
        self.x_dim=x_dim
        self.sample_size=min(100*x_dim,5000)
        self.sobol_engine=SobolEngine(self.x_dim,scramble=True)
        self.disturbance_probability=min(1,20/self.x_dim)

    def RAASP_sampling(self,
                       tr_x_best,
                       tr_lb,
                       tr_ub):
        
        sobol_sequence=self.sobol_engine.draw(self.sample_size)
        sobol_sequence=tr_lb+(tr_ub-tr_lb)*sobol_sequence
        mask=torch.rand_like(sobol_sequence)<=self.disturbance_probability
        ind=torch.where(mask.sum(dim=1)==0)[0]
        mask[ind,torch.randint(0,self.x_dim,size=(len(ind),))]=1
        tr_x_best=tr_x_best.unsqueeze(0)
        candidates_set=torch.where(mask,sobol_sequence,tr_x_best)
        return candidates_set


        


