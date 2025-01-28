import torch
import numpy as np
from typing import Tuple
import scipy.stats as stats
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


ETA = 2.0
VALUE_FN = "neg_linear" # "neg_linear", "linear", "squared"

class Net(torch.nn.Module):
    """
    A simple neural network that outputs a mean and log-variance as learnable parameters.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.mean = torch.nn.Parameter(torch.randn(1))
        self.logvar = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        batch_size = x.size(0)
        # Expand mean and logvar to match batch size
        mean = self.mean.expand(batch_size, 1)
        logvar = self.logvar.expand(batch_size, 1)
        return torch.cat([mean, logvar], dim=-1)

def generate_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates synthetic data and splits it into training and testing sets.
    Returns:
        x_train, y_train, x_test, y_test
    """
    x = torch.zeros(1000, 1)
    y = torch.randn(1000, 1)
    # Shuffle and split data
    idx = torch.randperm(len(x))
    x, y = x[idx], y[idx]
    x_train, y_train = x[:800], y[:800]
    x_test, y_test = x[800:], y[800:]
    return x_train, y_train, x_test, y_test

@torch.compile
def compute_loss(pred_mean: torch.Tensor, pred_logvars: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log-likelihood loss between predictions and targets.
    """
    inv_var = torch.exp(-pred_logvars)
    mse_loss = ((pred_mean - y) ** 2 * inv_var).mean()
    var_loss = pred_logvars.mean()
    return mse_loss + var_loss

def fit_nominal(
    network: torch.nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int = 100
) -> torch.nn.Module:
    """
    Trains the network using standard negative log-likelihood loss.
    """
    optimizer = torch.optim.SGD(network.parameters(), lr=0.0002)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    optimal_test_loss = float('inf')
    grace_period = 2000
    epochs_since_improvement = 0

    for epoch in range(5000):
        # Training loop
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred_mean, pred_logvars = network(x_batch).chunk(2, dim=-1)
            loss = compute_loss(pred_mean, pred_logvars, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        if epoch % 100 == 0:
            with torch.no_grad():
                pred_mean, pred_logvars = network(x_test).chunk(2, dim=-1)
                loss = compute_loss(pred_mean, pred_logvars, y_test)
                print(f"Epoch {epoch}, Test Loss: {loss.item():.4f}")
                if epoch > grace_period and loss > optimal_test_loss + 0.005:
                    break
                if loss < optimal_test_loss:
                    optimal_test_loss = loss
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                    if epochs_since_improvement > 10:
                        break

    return network

@torch.compile
def value_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Value function used in the robustness term.
    """
    x = x.clone()
    match VALUE_FN:
        case "neg_linear":
            return -x
        case "linear":
            return x
        case "squared":
            return x.pow(2)
        case _:
            raise NotImplementedError(f"Value function {VALUE_FN} not implemented")

@torch.compile
def compute_robust_loss(
    pred_mean: torch.Tensor,
    pred_logvars: torch.Tensor,
    y: torch.Tensor,
    nominal_mean: torch.Tensor,
    nominal_logvars: torch.Tensor
) -> torch.Tensor:
    """
    Computes the robust loss with a KL divergence term and a robustness regularization term.
    """
    pred_std = torch.exp(pred_logvars)
    nominal_std = torch.exp(nominal_logvars)

    pred_gaussian = torch.distributions.Normal(pred_mean, pred_std)
    nominal_gaussian = torch.distributions.Normal(nominal_mean, nominal_std)

    # KL divergence between predicted and nominal distributions
    kl_divergence = torch.distributions.kl_divergence(pred_gaussian, nominal_gaussian).mean()

    # Robustness term
    samples = pred_gaussian.sample()
    robust_loss = (pred_gaussian.log_prob(samples) * value_fn(0.99 * samples).detach()).mean()

    return kl_divergence + ETA * robust_loss

def fit_robust(
    network: torch.nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    nominal_net: torch.nn.Module,
    batch_size: int = 200
) -> torch.nn.Module:
    """
    Trains the network with a robustness regularization term.
    """
    optimizer = torch.optim.SGD(network.parameters(), lr=0.0002)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    nominal_net.eval()
    optimal_test_loss = float('inf')
    grace_period = 3000
    epochs_since_improvement = 0

    for epoch in range(10000):
        # Training loop
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred_mean, pred_logvars = network(x_batch).chunk(2, dim=-1)
            with torch.no_grad():
                nominal_mean, nominal_logvars = nominal_net(x_batch).chunk(2, dim=-1)
            loss = compute_robust_loss(
                pred_mean, pred_logvars, y_batch,
                nominal_mean.detach(), nominal_logvars.detach()
            )
            loss.backward()
            optimizer.step()

        # Validation
        if epoch % 100 == 0:
            with torch.no_grad():
                pred_mean, pred_logvars = network(x_test).chunk(2, dim=-1)
                loss = compute_loss(pred_mean, pred_logvars, y_test)
                print(f"Epoch {epoch}, Test Loss: {loss.item():.4f}")
                if epoch > grace_period and loss > optimal_test_loss + 0.005:
                    break
                if loss < optimal_test_loss:
                    optimal_test_loss = loss
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                    if epochs_since_improvement > 10:
                        break

    return network

def main():
    N_TRIALS = 10
    nominal_means, nominal_logvars = [], []
    robust_means, robust_logvars = [], []

    for _ in range(N_TRIALS):
        x_train, y_train, x_test, y_test = generate_data()

        # Train nominal network
        nominal_net = torch.compile(Net())
        nominal_net = fit_nominal(nominal_net, x_train, y_train, x_test, y_test)
        with torch.no_grad():
            nominal_mean, nominal_logvar = nominal_net(torch.zeros(1, 1)).chunk(2, dim=-1)

        # Train robust network
        robust_net = torch.compile(Net())
        robust_net = fit_robust(robust_net, x_train, y_train, x_test, y_test, nominal_net)
        with torch.no_grad():
            robust_mean, robust_logvar = robust_net(torch.zeros(1, 1)).chunk(2, dim=-1)

        # Store results
        nominal_means.append(nominal_mean.item())
        nominal_logvars.append(nominal_logvar.item())
        robust_means.append(robust_mean.item())
        robust_logvars.append(robust_logvar.item())

    # Compute mean and standard deviation for nominal and robust models
    nominal_mean = np.mean(nominal_means)
    nominal_std = np.mean(np.exp(0.5 * np.array(nominal_logvars)))
    robust_mean = np.mean(robust_means)
    robust_std = np.mean(np.exp(0.5 * np.array(robust_logvars)))

    # Print report
    print(f"Nominal Mean: {nominal_mean:.4f}, Nominal Std: {nominal_std:.4f}")
    print(f"Robust Mean: {robust_mean:.4f}, Robust Std: {robust_std:.4f}")

    # Append to toy_results.txt
    with open('toy_results.txt', 'a') as f:
        f.write(f"{VALUE_FN} {ETA}:\n")
        f.write(f"Nominal Mean: {nominal_mean:.4f}, Nominal Std: {nominal_std:.4f}\n")
        f.write(f"Robust Mean: {robust_mean:.4f}, Robust Std: {robust_std:.4f}\n\n")

    # Plot the two Gaussians
    x = np.linspace(-5, 5, 1000)
    nominal_pdf = stats.norm.pdf(x, nominal_mean, nominal_std)
    robust_pdf = stats.norm.pdf(x, robust_mean, robust_std)

    plt.plot(x, nominal_pdf, label='Nominal')
    plt.plot(x, robust_pdf, label='Robust')
    plt.legend()
    plt.title('Nominal vs Robust Gaussian')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.show()

if __name__ == "__main__":
    main()