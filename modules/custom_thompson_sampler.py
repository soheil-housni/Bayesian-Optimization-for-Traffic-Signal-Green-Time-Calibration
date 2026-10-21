import torch
from botorch.generation.sampling import MaxPosteriorSampling




class CustomThompsonSampler(MaxPosteriorSampling):
    def forward(self,
                X,
                num_samples: int = 1,
                observation_noise: bool = True):
        
        posterior = self.model.posterior(
            X,
            observation_noise=observation_noise,
            posterior_transform=self.posterior_transform,
        )
        samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
        max_sample,_=torch.max(samples,dim=1)
        return self.maximize_samples(X, samples, num_samples),max_sample.item()