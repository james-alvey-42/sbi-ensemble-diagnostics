# Global
import os
import time

import numpy as np
import sbibm
import torch
from sbi.inference.posteriors import DirectPosterior
from torch.optim import AdamW
from traitlets import observe


# Local
import sbi_ensemble_diagnostics.utils as ut

# ================== Settings some default parameters ================================ #
# Choose the example, should be in the list of examples available in sbibm
example_name = "gaussian_mixture"
# Path to the directory where data will be stored (will add n_train_data)
outpath = "KL_data/"
# Number of networks in the ensemble
n_networks = 3
# Batch size for training
def_batch_size = 100
# Whether to shuffle the data
def_shuffle = True
# Type of dataloader, alternatives are: "resample", "fixed" , 'regenerate'
which_dataloader = "resample"
# Maximum number of epochs for training
num_epochs = 1000
# Frequency (in epochs) of checking KL divergence
check_every = 10
# Initial learning rate
learning_rate_initial = 1e-2
# Learning rate decay factor
gamma = 0.9
# Frequency (in epochs) of updating the learning rate scheduler
update_scheduler_every = 15
# Size(s) of the training dataset, you can run this with multiple ensambles with
# different values for ntrain
n_train_data = [100]
# Size of the validation dataset
n_val_data = [2 * v for v in n_train_data]
# Number of samples drawn for KL evaluation
n_samples = 10_000
# Tolerance for the computation of the KL divergence
KL_tol = 1e-3
# Target value for the KL divergence to stop the training loop
KL_stop = 1e-3


# ================ Initializing some things for the run ============================== #
# Set the target observation (should be a number in 1 - 10)
num_observation = 1
# Set initial learning rates for all networks in the ensemble
lrs = [learning_rate_initial for _ in range(n_networks)]
# Set step sizes for the learning rate scheduler for all networks in the ensemble
step_sizes = [update_scheduler_every for _ in range(n_networks)]
# Learning rate decay factors for all networks in the ensemble
gammas = [gamma for _ in range(n_networks)]

# Build a list of combinations for which we want to test the KL divergence
combinations = []
for i in range(n_networks):
    for j in range(i + 1, n_networks):
        combinations.append((i, j))
        combinations.append((j, i))

# Total number of combinations to test
n_combinations = len(combinations)


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

    # Set up the task from the sbibm library
    task = sbibm.get_task(example_name)
    # Get the prior
    prior = task.get_prior_dist()
    # Set up the simulator
    simulator = task.get_simulator()
    # Pick an observation
    observation = task.get_observation(num_observation=num_observation)

    # Get the training data set
    train_data = ut.NPEData(
        num_samples=n_train_data,
        prior=prior,
        simulator=simulator,
        which_dataloader=which_dataloader,
    )
    # Get the validation data set
    val_data = ut.NPEData(
        num_samples=n_val_data,
        prior=prior,
        simulator=simulator,
        which_dataloader=which_dataloader,
    )
    # Define the training dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=def_batch_size, shuffle=def_shuffle
    )
    # Define the validation dataloader
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=def_batch_size, shuffle=def_shuffle
    )
    # Build density estimators for all networks in the ensemble
    density_estimators = [
        ut.build_density_estimator(
            num_samples=def_batch_size,
            prior=prior,
            simulator=simulator,
            which_dataloader=which_dataloader,
        )
        for _ in range(n_networks)
    ]
    # Set up optimizers for all the networks in the ensemble
    optimizers = [
        AdamW(density_estimator.parameters(), lr=lrs[i])
        for i, density_estimator in enumerate(density_estimators)
    ]
    # Set up schedulers for all networks in the ensemble
    schedulers = [
        ut.setup_scheduler(optimizers[i], step_size=step_sizes[i], gamma=gammas[i])
        for i in range(len(lrs))
    ]
    # List of lists to store the training losses for all networks in the ensemble
    train_losses = [[] for _ in range(n_networks)]
    # List of lists to store the validation losses for all networks in the ensemble
    val_losses = [[] for _ in range(n_networks)]
    # List to store the KL divergence values
    divergence = []

    # Initialize training epoch
    epoch = 0
    # Initialize a large value for the KL divergence
    dd = 10

    # Start time for the run
    t0 = time.perf_counter()

    # Here starts the training loop
    while dd > KL_stop and epoch < num_epochs:
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

        for network_id, scheduler in zip(range(n_networks), schedulers):
            epoch_val_loss[network_id] /= len(val_dataloader)
            val_losses[network_id].append(epoch_val_loss[network_id])
            scheduler.step()

        if (epoch % check_every == 0 and epoch > 1) or epoch == num_epochs - 1:

            print(f"Now at epoch = {epoch}", end=" ")
            posteriors = []
            KL_vals = np.zeros((n_combinations))

            for i in range(n_networks):
                posteriors.append(DirectPosterior(density_estimators[i], prior))

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

            dd = np.abs(np.max(divergence[-1]))

            print(
                "Max KL divergence = %.3f, total time = %.2f"
                % (dd, time.perf_counter() - t0)
            )

        epoch += 1

    # The 10 here are the number of available benchmarks in sbibm
    p_samples = np.zeros((10, n_networks, n_samples, observation.shape[-1]))
    # In sbibm the examples have 10_000 posterior samplesy
    reference_samples = np.zeros((10, 10_000, observation.shape[-1]))

    # Test how well we do for all observations
    for i in range(10):
        reference_samples[i] = task.get_reference_posterior_samples(
            num_observation=i + 1
        )
        observation = task.get_observation(num_observation=i + 1)

        for j in range(n_networks):
            p_samples[i, j] = np.array(
                posteriors[j].sample(
                    (n_samples,), x=observation, show_progress_bars=False
                )
            )

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


# A main function
if __name__ == "__main__":

    for i in range(len(n_train_data)):
        # Add the train dataset size to the path
        save_path = outpath + str(n_train_data[i]) + "/"

        # If the output directory does not exist create it
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        # Get the results
        res = run_inference(
            n_train_data=n_train_data[i],
            n_val_data=n_val_data[i],
            n_networks=n_networks,
            num_epochs=num_epochs,
            example_name=example_name,
            save_path=save_path,
            which_dataloader=which_dataloader,
        )
