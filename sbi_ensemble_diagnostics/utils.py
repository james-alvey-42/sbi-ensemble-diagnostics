import torch
import numpy as np
import matplotlib.pyplot as plt
from sbi.neural_nets.net_builders import build_nsf


def KLval(posterior_i, posterior_j, tol, n_samples, observation=None):
    KL_update, KL_old = 0.0, 0.0
    n_loops = 0
    while np.abs(KL_update - KL_old) > tol or n_loops == 0:
        n_loops += 1
        samples = posterior_i.sample(
            (n_samples,), x=observation, show_progress_bars=False
        )
        log_ratio = posterior_i.log_prob(samples, x=observation) - posterior_j.log_prob(
            samples, x=observation
        )
        KL_old = KL_update
        KL_update = (log_ratio.mean() + KL_old * (n_loops - 1)) / n_loops
    return KL_update


def KLval_2(posterior_i, posterior_j, tol, n_samples, kwargs1, kwargs2):
    KL_update, KL_old = 0.0, 0.0
    n_loops = 0
    while np.abs(KL_update - KL_old) > tol or n_loops == 0:
        n_loops += 1
        samples = posterior_i.rvs(**kwargs1, size=n_samples)
        log_ratio = np.log(posterior_i.pdf(samples, **kwargs1)) - np.log(
            posterior_j.pdf(samples, **kwargs2)
        )
        KL_old = KL_update
        KL_update = (np.mean(log_ratio) + KL_old * (n_loops - 1)) / n_loops

    return KL_update


class EmbeddingNet(torch.nn.Module):
    def __init__(self, in_features=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, 2),
        )

    def forward(self, x):
        return self.net(x)


class NPEData(torch.utils.data.Dataset):
    def __init__(
        self,
        num_samples: int,
        prior: torch.distributions.Distribution,
        simulator,
        which_dataloader="fixed",
        seed: int = 44,
    ):
        super().__init__()
        self.prior = prior
        self.simulator = simulator
        self.theta = prior.sample((num_samples,))
        self.x = simulator(self.theta)

        # Set the behavior of __getitem__ based on which_dataloader
        if which_dataloader == "fixed":
            self._getitem_fn = self._fixed_getitem
        elif which_dataloader == "resample":
            self._getitem_fn = self._resample_getitem
        elif which_dataloader == "regenerate":
            self._getitem_fn = self._regenerate_getitem
        else:
            raise ValueError("Invalid dataloader option.")

    def __len__(self):
        return self.theta.shape[0]

    def __getitem__(self, index: int):
        return self._getitem_fn(index)

    def _fixed_getitem(self, index: int):
        # Fixed behavior: return precomputed theta and x
        return self.theta[index], self.x[index]

    def _resample_getitem(self, index: int):
        # Resample behavior: resample x for the same theta
        theta = self.theta[index]
        x = self.simulator(theta)[0]
        return theta, x

    def _regenerate_getitem(self, index: int):
        # Resample behavior: regenerate theta and x
        theta = self.prior.sample((1,))
        x = self.simulator(theta)[0]
        return theta, x


def build_density_estimator(num_samples, prior, simulator, which_dataloader="fixed"):
    dummy_data = NPEData(
        num_samples=num_samples,
        prior=prior,
        simulator=simulator,
        which_dataloader=which_dataloader,
    )
    density_estimator = build_nsf(
        batch_x=dummy_data.theta,
        batch_y=dummy_data.x,
        input_dim=dummy_data.x.shape[-1],
    )
    return density_estimator


def setup_scheduler(optimizer, step_size, gamma=0.8):
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )
    return scheduler


def plot_losses(
    train_losses,
    val_losses,
    check_every,
    colors=[
        "#003f5c",
        "#d45087",
        "#f95d6a",
        "#ff7c43",
        "#ffa600",
    ],  # ["r", "g", "b", "y", "purple"],
    ax=None,
    plot_legend=True,
):

    if ax is None:
        plt.figure()
        ax = plt.gca()

    n_networks, n_train_losses = train_losses.shape
    _, num_epochs = val_losses.shape
    rescale = int(n_train_losses / num_epochs)
    x_vec = (np.arange(n_train_losses) + 1) / rescale
    xx_vec = np.arange(num_epochs) + 1

    for network_id in range(n_networks):
        ax.plot(
            x_vec,
            train_losses[network_id],
            c=colors[network_id],
            alpha=0.3,
            lw=0.5,
            label=f"Training loss (Network {network_id +1})",
        )
        ax.plot(
            xx_vec,
            val_losses[network_id],
            c=colors[network_id],
            lw=0.5,
            label=f"Validation loss (Network {network_id +1})",
            # marker="o",
        )
        # ax.set_ylim(-1, 7)

    # for i in range(0, int(num_epochs / check_every) + 1):
    #     ax.axvline(i * check_every, c="grey", linestyle="--", alpha=0.3)

    if plot_legend:
        ax.legend(fontsize=12)  # ncols=2)
