from scipy.stats.qmc import LatinHypercube
import torch
from modules.bo import InitialSamples


class TrustRegionInitialization():
    def __init__(self,
                  initial_sampler
                  ):
          
          self.initial_sampler=initial_sampler
          print(self.initial_sampler.objective_function.sumo_simulator.cfg_file)

    def initialize_trust_regions(self,
                                 sample_size,
                                 x_dim,
                                 x_bounds,
                                 n_trust_regions,
                                 num_restarts):
        
        all_trust_region_initial_x_norm=[]
        all_trust_region_initial_y=[]
        all_trust_region_initial_df=[]

        for rs in range(num_restarts):
          trust_region_initial_x_norm=[]
          trust_region_initial_y=[]
          trust_region_initial_df=[]
          for n in range(n_trust_regions):
               init_X_norm, init_Y, init_df=self.initial_sampler.get_initial_sample(
                    sample_size=sample_size,
                    x_dim=x_dim,
                    x_bounds=x_bounds
               )

               trust_region_initial_x_norm.append(init_X_norm)
               trust_region_initial_y.append(init_Y)
               trust_region_initial_df.append(init_df)

          all_trust_region_initial_x_norm.append(trust_region_initial_x_norm)
          all_trust_region_initial_y.append(trust_region_initial_y)
          all_trust_region_initial_df.append(trust_region_initial_df)

        return all_trust_region_initial_x_norm,all_trust_region_initial_y,all_trust_region_initial_df
        
        



