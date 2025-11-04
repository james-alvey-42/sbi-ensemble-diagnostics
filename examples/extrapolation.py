"""Extrapolation example: train flows on synthetic Gaussian-frequency data.

This example creates simple Gaussian 'frequency' data where the parameter
theta controls the mean. It trains an ensemble of normalizing-flow density
estimators (NSF) to learn p(theta | x) and periodically measures pairwise KL
divergences between ensemble members as a simple diagnostic of ensemble spread.

The script is intentionally small and documented with short comments explaining
the configuration options and the main steps (dataset, model, training loop,
KL evaluation and saving outputs).
"""

import os
import time

import numpy as np
import torch
from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets.net_builders import build_nsf
from torch.optim import AdamW


# Local
import sbi_ensemble_diagnostics.utils as ut


# ================== Settings / defaults ======================================
# Number of frequency samples per data point (controls input dimensionality)
n_freq_samples = 10
# Measurement noise standard deviation used when simulating data
obs_std = 0.01

# Ensemble and training settings
n_networks = 3
def_batch_size = 100
def_shuffle = True

num_epochs = 1000
check_every = 10
update_scheduler_every = 15
# List of training dataset sizes to run (can run multiple experiments in loop)
n_train_data = [500]
# Validation sizes (here simply double the train size)
n_val_data = [2 * v for v in n_train_data]
# Number of posterior samples used when estimating KL
n_samples = 10_000

# KL estimation tolerances and stop criteria
KL_tol = 1e-3
KL_stop = 1e-3
# Learning rates and scheduler multipliers (per-network lists)
lrs = [1e-2 for _ in range(n_networks)]
decreases = [0.9 for _ in range(n_networks)]


# Dataloader behaviour and output path
which_dataloader = "resample"  # alternatives: 'fixed', 'regenerate'
outpath = "extrapolation_data/"
# Simple uniform prior over theta in [-1, 1]
uniform_prior = torch.distributions.Uniform(-1, 1)


combinations = []
for i in range(n_networks):
    for j in range(i + 1, n_networks):
        combinations.append((i, j))
        combinations.append((j, i))

n_combinations = len(combinations)


# Create the output directory if missing
if not os.path.isdir(outpath):
    os.mkdir(outpath)


def get_estimates(data):
    """
    Get estimates from the data.

    Parameters:
    data (torch.Tensor): Input data.

    Returns:
    torch.Tensor: Estimates.
    """
    return torch.mean(data, dim=0), obs_std / np.sqrt(n_freq_samples)


def generate_gaussian_data(mean, std=obs_std):
    """
    Generate Gaussian data for testing.

    Parameters:
    mean (float): Mean of the Gaussian distribution.
    std (float): Standard deviation of the Gaussian distribution.
    n_samples (int): Number of samples to generate.

    Returns:
    torch.Tensor: Generated Gaussian data.
    """

    return torch.normal(mean, torch.full_like(mean, std))


def simulator(theta, std=obs_std, n_samples=n_freq_samples):
    """
    Simulates data with the correct shape for the density estimator.

    Parameters:
    theta (torch.Tensor): Input parameters.
    std (float): Standard deviation of the Gaussian noise.
    n_samples (int): Number of frequency samples.

    Returns:
    torch.Tensor: Simulated data with shape (batch_size, n_freq_samples).
    """

    mean_to_use = theta.unsqueeze(1).expand(
        -1, n_samples
    )  # Shape: (batch_size, n_samples)
    data = generate_gaussian_data(
        mean_to_use, std=std
    )  # Shape: (batch_size, n_samples)
    return data.squeeze()


def density_estimator_extrapolation(
    num_samples, prior, simulator, which_dataloader=which_dataloader
):
    dummy_data = NNPEData(
        num_samples=num_samples,
        prior=prior,
        simulator=simulator,
        which_dataloader=which_dataloader,
    )

    density_estimator = build_nsf(
        batch_x=dummy_data.theta,
        batch_y=dummy_data.x,
        input_dim=n_freq_samples,
        out_dim=1,
    )

    return density_estimator


class NNPEData(ut.NPEData):
    def _resample_getitem(self, index: int):
        # Resample behavior: resample x for the same theta
        theta = torch.atleast_1d(self.theta[index])  #
        x = self.simulator(theta)
        return theta, x


def my_KLval(posterior_i, posterior_j, tol, n_samples, observation=None):
    """
    Estimate KL(posterior_i || posterior_j) by Monte Carlo averaging.

    The KL is estimated by sampling from posterior_i and computing the
    expectation of log p_i - log p_j. To reduce estimator variance we perform
    repeated batches and average the running estimate until it stabilises within
    `tol` (or until at least one batch is collected).

    Parameters
    ----------
    posterior_i, posterior_j : sbi posterior-like objects
        Objects implementing .sample(...) and .log_prob(...).
    tol : float
        Convergence tolerance for the running average of the KL estimate.
    n_samples : int
        Number of samples to draw for each Monte Carlo batch.
    observation : array-like or torch.Tensor, optional
        Conditioning observation passed to the posterior methods.

    Returns
    -------
    float
        Estimated KL divergence.
    """

    KL_update, KL_old = 0.0, 0.0
    n_loops = 0

    # Repeat sampling until the running average changes by less than `tol`.
    while np.abs(KL_update - KL_old) > tol or n_loops == 0:
        n_loops += 1
        samples = posterior_i.sample(
            (n_samples,), x=observation, show_progress_bars=False
        )

        # Ensure samples have shape (n_samples, 1) expected by log_prob
        samples = samples.view(n_samples, 1)

        log_ratio = posterior_i.log_prob(samples, x=observation) - posterior_j.log_prob(
            samples, x=observation
        )
        KL_old = KL_update
        # Running average update to stabilise the estimate across batches
        KL_update = (log_ratio.mean() + KL_old * (n_loops - 1)) / n_loops

    return KL_update


observation_theta = uniform_prior.sample((1,))
observation_data = simulator(observation_theta)


def run_inference(
    n_networks=n_networks,
    n_train_data=n_train_data,
    n_val_data=n_val_data,
    num_epochs=num_epochs,
    which_dataloader=which_dataloader,
    save_path=outpath,
):
    """Run training for the extrapolation example.

    Contract / behaviour:
    - Inputs are lightweight integers/lists controlling dataset sizes and
      training behaviour. ``n_train_data`` and ``n_val_data`` may be scalars
      (int) or small lists; the top-level script calls this function for each
      requested train-set size.
    - Model training updates every network in the ensemble on each batch.
    - KL divergences are computed periodically using ``my_KLval`` and the
      maximum absolute KL is used as a simple convergence metric.
    - On KeyboardInterrupt (or at the end of training) the function saves
      network weights and numpy archives containing diagnostics and posterior
      samples.

    Notes on shapes and types
    -------------------------
    - ``theta`` tensors have shape ``(batch_size,)`` or ``(batch_size, 1)``
      depending on the dataset/dataloader behaviour. The density estimators
      expect ``x`` to have shape ``(batch_size, n_freq_samples)`` and produce
      samples of shape ``(n_samples, 1)`` when drawing posterior samples.

    Returns
    -------
    train_losses, val_losses, divergence
        Lists containing per-network training and validation loss histories
        and the KL divergence traces recorded at each check-point.
    """

    # Ensure output directory exists
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # Quick summary of chosen configuration
    print("Save path: ", save_path, "\n")
    print(f"Will use {n_networks} networks")
    print(f"Will use {n_train_data} training data")
    print(f"Will use {n_val_data} validation data\n")

    # --- Prepare datasets ---------------------------------------------------
    print("Generating training and validation data...", end=" ")

    train_data = NNPEData(
        num_samples=n_train_data,
        prior=uniform_prior,
        simulator=simulator,
        which_dataloader=which_dataloader,
    )

    val_data = NNPEData(
        num_samples=n_val_data,
        prior=uniform_prior,
        simulator=simulator,
        which_dataloader=which_dataloader,
    )
    print(" done")

    # --- DataLoader wrappers -----------------------------------------------
    print("Defining dataloaders...", end=" ")
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=def_batch_size, shuffle=def_shuffle
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=def_batch_size, shuffle=def_shuffle
    )
    print(" done")

    # --- Model construction -------------------------------------------------
    print("Defining density estimators...", end=" ")
    density_estimators = [
        density_estimator_extrapolation(
            num_samples=def_batch_size,
            prior=uniform_prior,
            simulator=simulator,
            which_dataloader=which_dataloader,
        )
        for _ in range(n_networks)
    ]
    print(" done")

    # --- Optimizers & schedulers -------------------------------------------
    print("Defining optimizers and schedulers...", end=" ")
    optimizers = [
        AdamW(density_estimator.parameters(), lr=lrs[i])
        for i, density_estimator in enumerate(density_estimators)
    ]

    schedulers = [
        ut.setup_scheduler(optimizers[i], step_size=check_every, gamma=decreases[i])
        for i in range(len(lrs))
    ]
    print(" done")

    # Containers for diagnostics
    train_losses = [[] for _ in range(n_networks)]
    val_losses = [[] for _ in range(n_networks)]
    divergence = []

    t0 = time.perf_counter()

    epoch = 0
    dd = 10

    # --- Training loop -----------------------------------------------------
    print("\nStarting training...")
    try:

        while dd > KL_stop and epoch < num_epochs:

            # Training step for each batch: update all networks in the ensemble
            for density_estimator in density_estimators:
                density_estimator.train()

            for theta, x in train_dataloader:
                network_id = 0
                for density_estimator, optimizer in zip(density_estimators, optimizers):
                    loss = density_estimator.loss(theta, x).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_losses[network_id].append(loss.item())
                    network_id += 1

            # Validation pass: compute mean validation loss per network
            for density_estimator in density_estimators:
                density_estimator.eval()

            epoch_val_loss = [0.0 for _ in range(n_networks)]

            with torch.no_grad():
                for theta, x in val_dataloader:
                    network_id = 0
                    for density_estimator in density_estimators:
                        epoch_val_loss[network_id] += (
                            density_estimator.loss(theta, x).mean().item()
                        )
                        network_id += 1

            # Step schedulers and record validation losses
            for network_id, scheduler in zip(range(n_networks), schedulers):
                epoch_val_loss[network_id] /= len(val_dataloader)
                val_losses[network_id].append(epoch_val_loss[network_id])
                scheduler.step()

            # Periodically evaluate KL divergences between ensemble members
            if (epoch % check_every == 0 and epoch > 1) or epoch == num_epochs - 1:

                print(
                    "Now at epoch = %d, total time = %.2fs"
                    % (epoch, time.perf_counter() - t0),
                )
                posteriors = []
                KL_vals = np.zeros((n_combinations))

                for i in range(n_networks):
                    posteriors.append(
                        DirectPosterior(density_estimators[i], uniform_prior)
                    )

                for k in range(len(combinations)):
                    i, j = combinations[k]
                    KL_vals[k] = my_KLval(
                        posteriors[i],
                        posteriors[j],
                        KL_tol,
                        n_samples,
                        observation=observation_data,
                    )

                divergence.append(KL_vals)

                KL_max = np.max(np.abs(divergence[-1]))
                KL_mean = np.mean(divergence[-1])
                KL_std = np.std(divergence[-1])

                print("KL divergence values:\n", divergence[-1])
                print(
                    "max KL = %.3f, mean KL = %.3f, std KL = %.3f\n"
                    % (KL_max, KL_mean, KL_std)
                )

            epoch += 1
        # raise an error to enter the except part
        raise KeyboardInterrupt

    except KeyboardInterrupt:
        # Save models and diagnostic outputs when interrupted
        print("\nEntering the storing part")
        # Ensure a subdirectory for network states exists (keeps outputs tidy)
        states_dir = os.path.join(save_path, "network_states")
        if not os.path.isdir(states_dir):
            os.mkdir(states_dir)

        for i in range(n_networks):
            torch.save(
                density_estimators[i].state_dict(),
                os.path.join(states_dir, f"network_{i}_{epoch}_{check_every}.pth"),
            )

        p_samples = np.zeros((n_networks, n_samples))
        for i in range(n_networks):
            p_samples[i] = np.array(
                posteriors[i].sample(
                    (n_samples,), x=observation_data, show_progress_bars=False
                )
            )

        np.savez(
            save_path + f"_{epoch}_{check_every}.npz",
            train=train_losses,
            val=val_losses,
            KL=np.array(divergence),
            post_samples=p_samples,
        )

        return train_losses, val_losses, divergence


if __name__ == "__main__":

    for i in range(len(n_train_data)):
        save_path = outpath + str(n_train_data[i]) + "/"

        if not os.path.isdir(save_path):
            os.mkdir(save_path)
            os.mkdir(save_path + "network_states/")

        res = run_inference(
            n_train_data=n_train_data[i],
            n_val_data=n_val_data[i],
            n_networks=n_networks,
            num_epochs=num_epochs,
            save_path=save_path,
            which_dataloader=which_dataloader,
        )
