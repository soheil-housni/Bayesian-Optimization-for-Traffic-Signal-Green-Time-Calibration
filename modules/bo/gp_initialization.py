from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel, Kernel, rbf_kernel
from gpytorch.constraints import Interval
from gpytorch.means import ConstantMean
import torch

#the kernels defined are used in this function to initialize the Gaussian Process model 


def initialize_gp_model(train_X,train_Y,initial_length_scale_value=0.6931):

    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))

    covar_module = ScaleKernel( MaternKernel(nu=2.5) )
    mean_module=ConstantMean()

    covar_module.base_kernel.lengthscale=initial_length_scale_value

    gp_model = SingleTaskGP(
        train_X, train_Y,
        mean_module=mean_module,
        covar_module=covar_module, likelihood=likelihood, 
        outcome_transform=Standardize(m=1))

    gp_model.covar_module.base_kernel.lengthscale=initial_length_scale_value
    gp_model.mean_module.constant=0
    gp_model.covar_module.outputscale=0.6931


    gp_mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    return gp_model, gp_mll