from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel, Kernel, rbf_kernel
from gpytorch.constraints import Interval
from gpytorch.means import ConstantMean
import torch
from .custom_kernel_functions import Linear_Regression, Only_fA_Linear_Regression
from .custom_mean_functions import CustomMean

#the kernels defined are used in this function to initialize the Gaussian Process model 

class InitaliseGPModel():
    def __init__(self,
                 x_bounds=None,
                 fA_function=None,
                 nu=2.5):
        self.x_bounds=x_bounds
        self.fA_function=fA_function
        self.nu=nu

    def initialize_gp_model(self,
                            train_X,
                            train_Y,
                            kernel_name=None,
                            mean_name=None,
                            initial_length_scale_value=0.6931,
                            Y_train_mean=None,
                            Y_train_std=None):

        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))

        if kernel_name is None or kernel_name=="MaternKernel_default":
            covar_module = ScaleKernel( MaternKernel(nu=self.nu) )
            

        elif kernel_name == "linear_regression":
            covar_module = ScaleKernel(Linear_Regression(bounds=self.x_bounds,fA_function=self.fA_function,raw_fA_lengthscale=0,nu=self.nu) )

        elif kernel_name=="only_fA_linear_regression":
            covar_module = ScaleKernel(Only_fA_Linear_Regression(bounds=self.x_bounds,fA_function=self.fA_function,nu=self.nu))

        else:
            covar_module = ScaleKernel( MaternKernel(nu=self.nu))

        if mean_name is None or mean_name=="ConstantMean":
            mean_module=ConstantMean()
        elif mean_name=="linear_regression":
            mean_module = CustomMean(bounds=self.x_bounds,fa_function=self.fA_function,Y_train_mean=Y_train_mean,Y_train_std=Y_train_std)
        else:
            mean_module=ConstantMean()

        gp_model = SingleTaskGP(
            train_X, train_Y,
            mean_module=mean_module,
            covar_module=covar_module, likelihood=likelihood, 
            outcome_transform=Standardize(m=1))

        covar_module.base_kernel.lengthscale=initial_length_scale_value
        gp_model.covar_module.base_kernel.lengthscale=initial_length_scale_value
        if mean_name is None or mean_name=="ConstantMean":
            gp_model.mean_module.constant=0
        gp_model.covar_module.outputscale=0.6931


        gp_mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        return gp_model, gp_mll