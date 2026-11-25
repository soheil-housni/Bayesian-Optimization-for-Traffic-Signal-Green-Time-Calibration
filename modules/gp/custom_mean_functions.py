from botorch.utils.transforms import unnormalize
import torch
from gpytorch.means.mean import Mean


class CustomMean(Mean):
    def __init__(self,
                 bounds,
                 fa_function,
                 Y_train_mean,
                 Y_train_std,
                  **kwargs):
        super().__init__()
        self.bounds=bounds
        self.fa_function=fa_function
        self.Y_train_mean=Y_train_mean
        self.Y_train_std=Y_train_std
            
    def forward(self, x):
        x=unnormalize(x,self.bounds)
        x=x.to(torch.double)
        x_fA = self.fa_function.fA(x)
        x_fA=-x_fA
        x_fA = x_fA.to(torch.double)
        x_fA=(x_fA-self.Y_train_mean)/(self.Y_train_std)
        return x_fA.squeeze(-1) 