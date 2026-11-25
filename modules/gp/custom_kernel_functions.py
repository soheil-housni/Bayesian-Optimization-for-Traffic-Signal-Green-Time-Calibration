import torch
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel, Kernel, rbf_kernel
from botorch.utils.transforms import unnormalize
import math
from gpytorch.functions import MaternCovariance
from gpytorch.settings import trace_mode


class Linear_Regression(Kernel):

    has_lengthscale = True

    def __init__(self,
                 bounds,
                 fA_function,
                 raw_fA_lengthscale=0.0,
                 rbf=False,
                 nu=None,
                 **kwargs):

        if nu:
            if nu not in {0.5, 1.5, 2.5}:
                raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        
        super(Linear_Regression, self).__init__(**kwargs)
        self.nu = nu
        self.bounds=bounds
        self.fA_function=fA_function
        self.register_parameter("raw_fA_lengthscale",torch.nn.Parameter(torch.tensor([raw_fA_lengthscale],dtype=torch.double)))
        self.rbf=rbf

    @property
    def fA_lengthscale(self):
        return torch.nn.functional.softplus(self.raw_fA_lengthscale)


    def forward(self, x1_init, x2_init, diag=False, **params):

        x1_init_unormalized=unnormalize(x1_init,self.bounds)
        x2_init_unormalized=unnormalize(x2_init,self.bounds)

        fA_x1=self.fA_function.fA(x1_init_unormalized)
        fA_x2=self.fA_function.fA(x2_init_unormalized)

        fA_distance = self.covar_dist(fA_x1.div(self.fA_lengthscale), fA_x2.div(self.fA_lengthscale), diag=diag, **params)

        x_distance=self.covar_dist(x1_init.div(self.lengthscale),x2_init.div(self.lengthscale))

        if self.nu:
            fA_exp_component = torch.exp(-math.sqrt(self.nu * 2) * fA_distance)
            x_exp_component = torch.exp(-math.sqrt(self.nu * 2) * x_distance)
                        
            if self.nu == 0.5:
                fA_constant_component = 1
                x_constant_component=1
            elif self.nu == 1.5:
                fA_constant_component = (math.sqrt(3) * fA_distance).add(1)
                x_constant_component= (math.sqrt(3) * x_distance).add(1)
            elif self.nu == 2.5:
                fA_constant_component = (math.sqrt(5) * fA_distance).add(1).add((5.0 / 3.0) * fA_distance**2)
                x_constant_component = (math.sqrt(5) * x_distance).add(1).add((5.0 / 3.0) * x_distance**2)


            fA_custom_component=fA_constant_component * fA_exp_component
            x_component=x_constant_component*x_exp_component

            return x_component*fA_custom_component

        if self.rbf:
            fA_custom_component=torch.exp((-fA_distance**2)/2)
            x_component=torch.exp((-x_distance**2)/2)
            return fA_custom_component*x_component




class Only_fA_Linear_Regression(Kernel):

    has_lengthscale = True

    def __init__(self,
                 bounds,
                 fA_function,
                 rbf=False,
                 nu=None,
                 **kwargs):

        if nu:
            if nu not in {0.5, 1.5, 2.5}:
                raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        
        super().__init__(**kwargs)
        self.nu = nu
        self.bounds=bounds
        self.fA_function=fA_function
        self.rbf=rbf


    def forward(self, x1_init, x2_init, diag=False, **params):


        x1_init_unormalized=unnormalize(x1_init,self.bounds)
        x2_init_unormalized=unnormalize(x2_init,self.bounds)

        fA_x1=self.fA_function.fA(x1_init_unormalized)
        fA_x2=self.fA_function.fA(x2_init_unormalized)

        fA_x1=-fA_x1
        fA_x2=-fA_x2

        fA_distance = self.covar_dist(fA_x1.div(self.lengthscale), fA_x2.div(self.lengthscale), diag=diag, **params)


        if self.nu:
            fA_exp_component = torch.exp(-math.sqrt(self.nu * 2) * fA_distance)
                        
            if self.nu == 0.5:
                fA_constant_component = 1
            elif self.nu == 1.5:
                fA_constant_component = (math.sqrt(3) * fA_distance).add(1)
            elif self.nu == 2.5:
                fA_constant_component = (math.sqrt(5) * fA_distance).add(1).add((5.0 / 3.0) * fA_distance**2)


            fA_custom_component=fA_constant_component * fA_exp_component

            return fA_custom_component

        if self.rbf:
            fA_custom_component=torch.exp((-fA_distance**2)/2)
            return fA_custom_component




