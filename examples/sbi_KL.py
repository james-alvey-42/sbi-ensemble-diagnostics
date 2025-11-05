"""Example script: train an ensemble and compute KL divergences.

This script is a runnable example used in development and for small-scale
experiments. It trains an ensemble of normalizing-flow density estimators on a
chosen sbibm benchmark, periodically computes pairwise Kullback-Leibler (KL)
divergences between the learned posteriors, and saves losses, KL traces and
posterior samples to disk.

The file is intentionally lightweight and documented with short comments that
explain the configuration options and the main steps taken in the training loop.
"""

# Global imports
import os
import time

import numpy as np
import sbibm
import torch
from sbi.inference.posteriors import DirectPosterior
from torch.optim import AdamW


# Local utilities from this package
import sbi_ensemble_diagnostics.utils as ut

example_name = "gaussian_mixture"
# Base output path where results will be stored; the train-set size is appended
outpath = "KL_data/"
# Number of networks in the ensemble
n_networks = 3
# Training batch size
def_batch_size = 100
# Shuffle training/validation data
def_shuffle = True
# Dataloader behavior: 'fixed' uses a fixed dataset, 'resample' resamples x for
# the same theta on every access, 'regenerate' draws new theta,x pairs each access.
which_dataloader = "resample"
# Maximum training epochs
num_epochs = 1000
# How often (in epochs) we compute KL divergences and print progress
check_every = 10
# Initial learning rate
learning_rate_initial = 1e-2
# Multiplicative decay factor for the LR scheduler
gamma = 0.9
# LR scheduler step size (epochs)
update_scheduler_every = 15
# List of training dataset sizes to run (can run multiple experiments in loop)
n_train_data = [100]
# Validation sizes (here simply double the train size)
n_val_data = [2 * v for v in n_train_data]
# Number of samples to draw when estimating KL between two posteriors
n_samples = 10_000
# Tolerance and stop criteria for iterative KL estimation
KL_tol = 1e-3
KL_stop = 1e-3


# ================ Derived / run-time defaults ==================================== #
# Which observation index to target for the training run (sbibm examples have
# multiple possible observations; these are 1-indexed in sbibm)
num_observation = 1
# Per-network initial learning rates (can be customised per network if needed)
lrs = [learning_rate_initial for _ in range(n_networks)]
# Scheduler step sizes for each network
step_sizes = [update_scheduler_every for _ in range(n_networks)]
# Scheduler gammas for each network
gammas = [gamma for _ in range(n_networks)]

# Build a list of ordered pair combinations between networks for KL checks.
# We include both (i,j) and (j,i) to compute KL(p_i||p_j) and KL(p_j||p_i).
combinations = []
for i in range(n_networks):
    for j in range(i + 1, n_networks):
        combinations.append((i, j))
        combinations.append((j, i))

# Number of KL pairs to evaluate
n_combinations = len(combinations)

# Create the output directory if missing
if not os.path.isdir(outpath):
    os.mkdir(outpath)


# ========================== Main training loop ====================================== #
def run_inference(
    n_networks=n_networks,
    n_train_data=n_train_data,
    n_val_data=n_val_data,
    num_epochs=num_epochs,
    example_name="gaussian_mixture",
    num_observation=num_observation,
    which_dataloader=which_dataloader,
    save_path=outpath,
):
    """
    Train an ensemble of density estimators and periodically compute pairwise KL
    divergences between the learned posteriors.

    Parameters
    ----------
    n_networks : int
        Number of models in the ensemble.
    n_train_data, n_val_data : int
        Number of samples for training/validation datasets.
    num_epochs : int
        Maximum number of training epochs.
    example_name : str
        sbibm benchmark name.
    num_observation : int
        Which sbibm observation index to use (1-indexed).
    which_dataloader : str
        Controls how dataset __getitem__ behaves (see utils.NPEData).
    save_path : str
        Directory where outputs will be written.

    Returns
    -------
    train_losses, val_losses, divergence
        Recorded training/validation losses and KL divergence history.
    """

    # --- Setup sbibm task and simulator/prior ---------------------------------
    task = sbibm.get_task(example_name)
    prior = task.get_prior_dist()
    simulator = task.get_simulator()
    # Load a single observation used for conditional inference during KL checks
    observation = task.get_observation(num_observation=num_observation)

    # --- Construct datasets and dataloaders ----------------------------------
    train_data = ut.NPEData(
        num_samples=n_train_data,
        prior=prior,
        simulator=simulator,
        which_dataloader=which_dataloader,
    )
    val_data = ut.NPEData(
        num_samples=n_val_data,
        prior=prior,
        simulator=simulator,
        which_dataloader=which_dataloader,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=def_batch_size, shuffle=def_shuffle
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=def_batch_size, shuffle=def_shuffle
    )

    # --- Build ensemble density estimators and optimizers --------------------
    density_estimators = [
        ut.build_density_estimator(
            num_samples=def_batch_size,
            prior=prior,
            simulator=simulator,
            which_dataloader=which_dataloader,
        )
        for _ in range(n_networks)
    ]

    # One optimizer per network (AdamW here)
    optimizers = [
        AdamW(density_estimator.parameters(), lr=lrs[i])
        for i, density_estimator in enumerate(density_estimators)
    ]

    # Learning-rate schedulers for each network
    schedulers = [
        ut.setup_scheduler(optimizers[i], step_size=step_sizes[i], gamma=gammas[i])
        for i in range(len(lrs))
    ]

    # Containers to record losses and KL values
    train_losses = [[] for _ in range(n_networks)]
    val_losses = [[] for _ in range(n_networks)]
    divergence = []

    # Initialize counters and a large starting KL
    epoch = 0
    dd = 10
    t0 = time.perf_counter()

    try:
        # --- Training loop -------------------------------------------------------
        # Loop until KL converges below KL_stop or until num_epochs
        while dd > KL_stop and epoch < num_epochs:
            # Set networks into training mode
            for density_estimator in density_estimators:
                density_estimator.train()

            # Iterate over training batches and update each network in the ensemble
            for theta, x in train_dataloader:
                network_id = 0
                for density_estimator, optimizer in zip(density_estimators, optimizers):
                    loss = density_estimator.loss(theta, x).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_losses[network_id].append(loss.item())
                    network_id += 1

            # Evaluation mode for validation
            for density_estimator in density_estimators:
                density_estimator.eval()

            # Compute mean validation loss per network for this epoch
            epoch_val_loss = [0.0 for _ in range(n_networks)]
            with torch.no_grad():
                for theta, x in val_dataloader:
                    network_id = 0
                    for density_estimator in density_estimators:
                        epoch_val_loss[network_id] += (
                            density_estimator.loss(theta, x).mean().item()
                        )
                        network_id += 1

            # Average validation losses and step schedulers
            for network_id, scheduler in zip(range(n_networks), schedulers):
                epoch_val_loss[network_id] /= len(val_dataloader)
                val_losses[network_id].append(epoch_val_loss[network_id])
                scheduler.step()

            # Periodically compute KL divergences between ensemble members
            if (epoch % check_every == 0 and epoch > 1) or epoch == num_epochs - 1:
                print(f"Now at epoch = {epoch},", end=" ")
                posteriors = []
                KL_vals = np.zeros((n_combinations))

                # Wrap the trained density estimators as sbi DirectPosterior objects
                for i in range(n_networks):
                    posteriors.append(DirectPosterior(density_estimators[i], prior))

                # Compute pairwise KLs using the utility function KLval
                for k in range(len(combinations)):
                    i, j = combinations[k]
                    KL_vals[k] = ut.KLval(
                        posteriors[i],
                        posteriors[j],
                        KL_tol,
                        n_samples,
                        observation=observation,
                    )

                divergence.append(KL_vals)
                # Use the maximum observed KL (absolute) as the convergence metric
                dd = np.abs(np.max(divergence[-1]))

                print(
                    "Max KL divergence = %.3f, total time = %.2f"
                    % (dd, time.perf_counter() - t0)
                )

            epoch += 1

        # raise an error to enter the except part
        raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\nEntering the storing part")
        # --- After training: save the network weights and draw posterior samples for
        # all benchmark observations ----
        for i in range(n_networks):
            torch.save(
                density_estimators[i].state_dict(),
                save_path + f"network_states/network_{i}_{epoch}_{check_every}.pth",
            )

        # sbibm provides 10 observations/benchmarks for each task; allocate arrays
        p_samples = np.zeros((10, n_networks, n_samples, observation.shape[-1]))
        # Reference posterior samples provided by sbibm (10k samples each)
        reference_samples = np.zeros((10, 10_000, observation.shape[-1]))

        for i in range(10):
            # Load reference posterior samples and corresponding observation
            reference_samples[i] = task.get_reference_posterior_samples(
                num_observation=i + 1
            )
            observation = task.get_observation(num_observation=i + 1)

            # Draw samples from each trained posterior for this observation
            for j in range(n_networks):
                p_samples[i, j] = np.array(
                    posteriors[j].sample(
                        (n_samples,), x=observation, show_progress_bars=False
                    )
                )

        # Save outputs (loss histories, KL traces, posterior samples and references)
        outname = save_path + example_name + f"_{epoch}_{check_every}.npz"
        print("I will store the output in: ", outname)

        np.savez(
            outname,
            train=train_losses,
            val=val_losses,
            KL=np.array(divergence),
            posterior_samples=p_samples,
            reference_samples=reference_samples,
        )

        return train_losses, val_losses, divergence


# A main function: run experiments for each requested training size
if __name__ == "__main__":

    for i in range(len(n_train_data)):
        # Append the train-set size to the base output path
        save_path = outpath + str(n_train_data[i]) + "/"

        # Create the output directory if missing
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
            os.mkdir(save_path + "network_states/")

        # Run the experiment and get recorded results
        res = run_inference(
            n_train_data=n_train_data[i],
            n_val_data=n_val_data[i],
            n_networks=n_networks,
            num_epochs=num_epochs,
            example_name=example_name,
            save_path=save_path,
            which_dataloader=which_dataloader,
        )
