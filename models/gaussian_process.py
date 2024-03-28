import numpy as np
import torch
import gpytorch
from .model import Model

class GaussianProcessModel(gpytorch.models.ExactGP, Model):
    """
    A simple Gaussian process model class for Bayesian optimization.
    """

    TRAINING_ITERATIONS = 200
    EARLY_STOPPING_CONSTANT_ITERATIONS = 20

    def __init__(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(None, None, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.name = "Gaussian Process"

    def forward(self, x):
        """
        Internal method to compute the forward pass of the Gaussian process model.
        :param x: Data points to make predictions for.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit(self, train_x: np.ndarray, train_y: np.ndarray):
        """
        Fit the Gaussian process model to the provided data.
        :param train_x: Training data points (input space: n x d).
        :param train_y: Training targets (output space: n x 1).
        """
        ### Convert inputs to torch tensors
        train_inputs = torch.from_numpy(train_x).float()
        train_targets = torch.from_numpy(train_y).float()
        self.set_train_data(train_inputs, train_targets, strict=False)
        ### Set model to training mode
        self.train()
        self.likelihood.train()
        ### Initialize optimizer and MLL
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        ### Run iterative optimization
        loss_history = []
        for i in range(GaussianProcessModel.TRAINING_ITERATIONS):
            optimizer.zero_grad()
            output = self(train_inputs)
            loss = -mll(output, self.train_targets)
            loss.backward()
            loss_history.append(loss.item())
            if i > GaussianProcessModel.EARLY_STOPPING_CONSTANT_ITERATIONS:
                history = loss_history[-GaussianProcessModel.EARLY_STOPPING_CONSTANT_ITERATIONS:]
                if torch.all(torch.isclose(torch.tensor(history), torch.tensor(loss_history[-1]))):
                    break

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with the trained Gaussian process model.
        :param test_x: Data points to make predictions for (n x d).
        :return: Tuple of mean and standard deviation of the predictions (n x 1).
        """
        ### Convert input to torch tensor
        x = torch.from_numpy(test_x).float()
        ### Set model to evaluation mode
        self.eval()
        self.likelihood.eval()
        ### Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self(x))
            return observed_pred.mean.numpy(), observed_pred.stddev.numpy()
