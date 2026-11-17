"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

BANDIT_REWARD = [1, 2, 3, 4]
NUMBER_OF_TRIALS = 20000
EPSILON = 5.0
PRECISION = 1.0
PRIOR_MEAN = 0.0
PRIOR_PRECISION = 0.01
SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


class Bandit(ABC):
    """
    Abstract multi-armed bandit experiment.

    A concrete subclass is one learning algorithm playing against a set of arms
    (advertisements). It must choose an arm and observe its reward (``pull``),
    learn from that reward (``update``), run a whole experiment (``experiment``)
    and summarise it (``report``).
    """
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        """
        :param p: true mean reward of every arm
        :type p: list[float]
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        :return: an unambiguous description of the algorithm and its settings
        :rtype: str
        """
        pass

    @abstractmethod
    def pull(self):
        """
        Choose an arm and observe the reward it pays.

        :return: index of the chosen arm and the observed reward
        :rtype: tuple[int, float]
        """
        pass

    @abstractmethod
    def update(self):
        """
        Learn from the reward observed after the last pull.
        """
        pass

    @abstractmethod
    def experiment(self):
        """
        Run the full experiment: repeat pull and update for every trial.
        """
        pass

    @abstractmethod
    def report(self):
        """
        Store the rewards in a CSV file and log the reward and regret.
        """
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

    def _init_arms(self, p, precision, seed):
        """
        Set up the arms, the random generator and the (empty) history.

        :param p: true mean reward of every arm
        :type p: list[float]
        :param precision: precision (1 / variance) of the reward noise
        :type precision: float
        :param seed: seed of the random generator, for reproducibility
        :type seed: int or None
        """
        self.p = np.asarray(p, dtype=float)
        self.precision = precision
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(len(self.p), dtype=int)
        self.chosen_arms = np.empty(0, dtype=int)
        self.rewards = np.empty(0)
        self.estimate_history = np.empty((0, len(self.p)))

    @property
    def name(self):
        """
        :return: name of the algorithm, as written to the CSV file
        :rtype: str
        """
        return type(self).__name__

    def _draw(self, arm):
        """
        Show ad ``arm`` once and return the reward it pays.

        :param arm: index of the arm
        :type arm: int
        :return: reward drawn from N(p[arm], 1 / precision)
        :rtype: float
        """
        return self.rng.normal(self.p[arm], 1 / np.sqrt(self.precision))

    def _run(self, n_trials):
        """
        Play ``n_trials`` rounds of pull and update, recording after each round
        the chosen arm, the reward and the algorithm's estimate of every arm.

        :param n_trials: number of trials
        :type n_trials: int
        """
        chosen_arms = np.empty(n_trials, dtype=int)
        rewards = np.empty(n_trials)
        estimate_history = np.empty((n_trials, len(self.p)))
        for i in range(n_trials):
            arm, reward = self.pull()
            self.update(arm, reward)
            chosen_arms[i], rewards[i] = arm, reward
            estimate_history[i] = self.estimates
        self.chosen_arms, self.rewards, self.estimate_history = chosen_arms, rewards, estimate_history

    def cumulative_rewards(self):
        """
        :return: total reward collected after each trial
        :rtype: numpy.ndarray
        """
        return np.cumsum(self.rewards)

    def cumulative_regrets(self):
        """
        Regret of a trial is the gap between the best arm's mean and the chosen
        arm's mean, so it measures the cost of the choices, not of the noise.

        :return: total regret accumulated after each trial
        :rtype: numpy.ndarray
        """
        return np.cumsum(self.p.max() - self.p[self.chosen_arms])

    def _report(self, csv_path, append):
        """
        Write one ``Bandit, Reward, Algorithm`` row per trial and log the
        cumulative and average reward and regret.

        :param csv_path: CSV file to write to
        :type csv_path: str or pathlib.Path
        :param append: append to an existing file instead of overwriting it
        :type append: bool
        """
        rows = pd.DataFrame({"Bandit": self.chosen_arms, "Reward": self.rewards, "Algorithm": self.name})
        rows.to_csv(csv_path, mode="a" if append else "w", header=not append, index=False)
        logger.debug(f"{self.name}: wrote {len(rows)} rows to {csv_path}")

        n_trials = len(self.rewards)
        total_reward = self.cumulative_rewards()[-1]
        total_regret = self.cumulative_regrets()[-1]
        logger.info(f"{self.name}: cumulative reward = {total_reward:,.2f} "
                    f"(average {total_reward / n_trials:.4f} per trial, best possible {self.p.max():g})")
        logger.info(f"{self.name}: cumulative regret = {total_regret:,.2f} "
                    f"(average {total_regret / n_trials:.4f} per trial)")
        logger.info(f"{self.name}: trials per bandit = {self.counts.tolist()}, "
                    f"estimated means = {np.round(self.estimates, 3).tolist()}")

#--------------------------------------#


class Visualization():
    """
    Plots of finished bandit experiments, saved as PNG files.
    """

    def __init__(self, output_dir=OUTPUT_DIR):
        """
        :param output_dir: folder the figures are saved to (created if missing)
        :type output_dir: str or pathlib.Path
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, fig, filename):
        """
        Save a figure to the output folder and close it.

        :param fig: figure to save
        :type fig: matplotlib.figure.Figure
        :param filename: name of the PNG file
        :type filename: str
        :return: path of the saved file
        :rtype: pathlib.Path
        """
        path = self.output_dir / filename
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {path}")
        return path

    def plot1(self, bandit):
        """
        Visualize the learning process of one algorithm, on a linear and a log
        trial axis: the running average reward against the best ad's mean (top)
        and the algorithm's estimate of every ad's mean against the truth (bottom).

        :param bandit: an algorithm that has run its experiment
        :type bandit: Bandit
        :return: path of the saved figure
        :rtype: pathlib.Path
        """
        # Visualize the performance of each bandit: linear and log
        trials = np.arange(1, len(bandit.rewards) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col")
        for col, scale in enumerate(["linear", "log"]):
            ax = axes[0, col]
            ax.plot(trials, bandit.cumulative_rewards() / trials, label="running average reward")
            ax.axhline(bandit.p.max(), color="black", linestyle="--", label="best ad's mean")
            ax.set_title(f"Average reward ({scale} scale)")
            ax.set_ylabel("average reward")
            ax.legend(loc="lower right")

            ax = axes[1, col]
            for arm, mean in enumerate(bandit.p):
                line, = ax.plot(trials, bandit.estimate_history[:, arm], label=f"Bandit {arm} (true mean {mean:g})")
                ax.axhline(mean, color=line.get_color(), linestyle=":")
            ax.set_title(f"Estimated mean of each bandit ({scale} scale)")
            ax.set_xlabel("trial")
            ax.set_ylabel("estimated mean")
            ax.legend(loc="lower right", fontsize="small")
            ax.set_xscale(scale)
        fig.suptitle(f"Learning process: {bandit!r}")
        fig.tight_layout()
        return self.save(fig, f"plot1_{bandit.name}.png")

    def plot2(self, bandits):
        """
        Compare the cumulative rewards and the cumulative regrets of several
        algorithms run on the same bandits. The rewards are drawn on log-log
        axes, since on linear axes the curves overlap once both algorithms have
        settled on the best ad.

        :param bandits: algorithms that have run their experiments
        :type bandits: list[Bandit]
        :return: path of the saved figure
        :rtype: pathlib.Path
        """
        # Compare E-greedy and thompson sampling cummulative rewards
        # Compare E-greedy and thompson sampling cummulative regrets
        trials = np.arange(1, len(bandits[0].rewards) + 1)
        fig, (ax_reward, ax_regret) = plt.subplots(1, 2, figsize=(13, 5))
        ax_reward.plot(trials, trials * bandits[0].p.max(), color="black", linestyle="--",
                       label="always the best ad (expected)")
        for bandit in bandits:
            ax_reward.plot(trials, bandit.cumulative_rewards(), label=bandit.name)
            ax_regret.plot(trials, bandit.cumulative_regrets(), label=bandit.name)
        ax_reward.set_xscale("log")
        ax_reward.set_yscale("log")
        ax_reward.set_title("Cumulative reward (log-log scale)")
        ax_reward.set_ylabel("cumulative reward")
        ax_regret.set_title("Cumulative regret")
        ax_regret.set_ylabel("cumulative regret")
        for ax in (ax_reward, ax_regret):
            ax.set_xlabel("trial")
            ax.legend()
        fig.tight_layout()
        return self.save(fig, "plot2_cumulative_reward_and_regret.png")

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy with epsilon decaying as 1/t.

    At trial t a random ad is shown with probability epsilon_t = min(1, epsilon / t);
    otherwise the ad with the highest sample-mean reward so far is shown. With
    epsilon > 1 the first ``epsilon`` trials are pure exploration.
    """

    def __init__(self, p, epsilon=EPSILON, precision=PRECISION, seed=None):
        """
        :param p: true mean reward of every ad
        :type p: list[float]
        :param epsilon: initial exploration rate; at trial t it is epsilon / t
        :type epsilon: float
        :param precision: precision (1 / variance) of the reward noise
        :type precision: float
        :param seed: seed of the random generator, for reproducibility
        :type seed: int or None
        """
        self._init_arms(p, precision, seed)
        self.epsilon = epsilon
        self.t = 0
        self.n_explored = 0
        self.estimates = np.zeros(len(self.p))

    def __repr__(self):
        """
        :return: the algorithm and its settings
        :rtype: str
        """
        return f"EpsilonGreedy(p={self.p.tolist()}, epsilon={self.epsilon:g}/t, precision={self.precision:g})"

    def pull(self):
        """
        Explore a random ad with probability epsilon / t, otherwise exploit the
        ad with the highest estimated mean, and observe its reward.

        :return: index of the chosen ad and the observed reward
        :rtype: tuple[int, float]
        """
        self.t += 1
        if self.rng.random() < self.epsilon / self.t:
            self.n_explored += 1
            arm = int(self.rng.integers(len(self.p)))
        else:
            arm = int(np.argmax(self.estimates))
        return arm, self._draw(arm)

    def update(self, arm, reward):
        """
        Update the sample mean of the chosen ad incrementally.

        :param arm: index of the ad that was shown
        :type arm: int
        :param reward: reward it paid
        :type reward: float
        """
        self.counts[arm] += 1
        self.estimates[arm] += (reward - self.estimates[arm]) / self.counts[arm]

    def experiment(self, n_trials=NUMBER_OF_TRIALS):
        """
        Run ``n_trials`` trials. Call it once per instance.

        :param n_trials: number of trials
        :type n_trials: int
        :return: this instance, so the call can be chained
        :rtype: EpsilonGreedy
        """
        logger.info(f"Running {self!r} for {n_trials:,} trials")
        self._run(n_trials)
        logger.debug(f"{self.name}: explored in {self.n_explored} of {n_trials:,} trials, "
                     f"final epsilon = {self.epsilon / self.t:.2e}")
        return self

    def report(self, csv_path=OUTPUT_DIR / "rewards.csv", append=False):
        """
        Store the rewards in a CSV file and log the reward and regret.

        :param csv_path: CSV file to write to
        :type csv_path: str or pathlib.Path
        :param append: append to an existing file instead of overwriting it
        :type append: bool
        """
        self._report(csv_path, append)

#--------------------------------------#

class ThompsonSampling(Bandit):
    """
    Thompson Sampling for Gaussian rewards with known precision tau.

    The mean of every ad has a Normal prior N(m0, 1 / lambda0). With tau known,
    the posterior after n rewards summing to s is again Normal, with

        lambda_n = lambda0 + n * tau
        m_n      = (lambda0 * m0 + tau * s) / lambda_n

    Each trial draws one sample from every ad's posterior and shows the ad with
    the largest sample.
    """

    def __init__(self, p, precision=PRECISION, prior_mean=PRIOR_MEAN, prior_precision=PRIOR_PRECISION, seed=None):
        """
        :param p: true mean reward of every ad
        :type p: list[float]
        :param precision: known precision tau (1 / variance) of the reward noise
        :type precision: float
        :param prior_mean: prior mean m0 of every ad's mean reward
        :type prior_mean: float
        :param prior_precision: prior precision lambda0 of every ad's mean reward
        :type prior_precision: float
        :param seed: seed of the random generator, for reproducibility
        :type seed: int or None
        """
        self._init_arms(p, precision, seed)
        self.prior_mean = prior_mean
        self.prior_precision = prior_precision
        self.sum_rewards = np.zeros(len(self.p))
        self.posterior_precision = np.full(len(self.p), float(prior_precision))
        self.estimates = np.full(len(self.p), float(prior_mean))

    def __repr__(self):
        """
        :return: the algorithm and its settings
        :rtype: str
        """
        return (f"ThompsonSampling(p={self.p.tolist()}, precision={self.precision:g}, "
                f"prior=N({self.prior_mean:g}, 1/{self.prior_precision:g}))")

    def pull(self):
        """
        Sample a mean for every ad from its posterior, show the ad with the
        largest sample and observe its reward.

        :return: index of the chosen ad and the observed reward
        :rtype: tuple[int, float]
        """
        samples = self.rng.normal(self.estimates, 1 / np.sqrt(self.posterior_precision))
        arm = int(np.argmax(samples))
        return arm, self._draw(arm)

    def update(self, arm, reward):
        """
        Update the Normal posterior of the chosen ad's mean.

        :param arm: index of the ad that was shown
        :type arm: int
        :param reward: reward it paid
        :type reward: float
        """
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward
        self.posterior_precision[arm] = self.prior_precision + self.counts[arm] * self.precision
        self.estimates[arm] = ((self.prior_precision * self.prior_mean + self.precision * self.sum_rewards[arm])
                               / self.posterior_precision[arm])

    def experiment(self, n_trials=NUMBER_OF_TRIALS):
        """
        Run ``n_trials`` trials. Call it once per instance.

        :param n_trials: number of trials
        :type n_trials: int
        :return: this instance, so the call can be chained
        :rtype: ThompsonSampling
        """
        logger.info(f"Running {self!r} for {n_trials:,} trials")
        self._run(n_trials)
        logger.debug(f"{self.name}: posterior standard deviations = "
                     f"{np.round(1 / np.sqrt(self.posterior_precision), 4).tolist()}")
        return self

    def report(self, csv_path=OUTPUT_DIR / "rewards.csv", append=False):
        """
        Store the rewards in a CSV file and log the reward and regret.

        :param csv_path: CSV file to write to
        :type csv_path: str or pathlib.Path
        :param append: append to an existing file instead of overwriting it
        :type append: bool
        """
        self._report(csv_path, append)




def comparison(bandits, output_dir=OUTPUT_DIR):
    """
    Compare the algorithms visually by where they spent their trials: the number
    of trials given to each ad, and how the share given to the best ad grows
    over time. Also logs a one-line summary per algorithm.

    :param bandits: algorithms that have run their experiments on the same ads
    :type bandits: list[Bandit]
    :param output_dir: folder the figure is saved to
    :type output_dir: str or pathlib.Path
    :return: path of the saved figure
    :rtype: pathlib.Path
    """
    # think of a way to compare the performances of the two algorithms VISUALLY and
    n_arms = len(bandits[0].p)
    best_arm = int(np.argmax(bandits[0].p))
    trials = np.arange(1, len(bandits[0].rewards) + 1)
    width = 0.8 / len(bandits)

    fig, (ax_counts, ax_best) = plt.subplots(1, 2, figsize=(13, 5))
    for i, bandit in enumerate(bandits):
        counts = np.bincount(bandit.chosen_arms, minlength=n_arms)
        bars = ax_counts.bar(np.arange(n_arms) + (i - (len(bandits) - 1) / 2) * width, counts, width,
                             label=bandit.name)
        ax_counts.bar_label(bars, fontsize="small")
        best_rate = np.cumsum(bandit.chosen_arms == best_arm) / trials
        ax_best.plot(trials, best_rate, label=bandit.name)
        logger.info(f"{bandit.name}: showed the best ad (Bandit {best_arm}) in "
                    f"{counts[best_arm] / len(bandit.chosen_arms):.2%} of trials, "
                    f"cumulative regret {bandit.cumulative_regrets()[-1]:,.2f}")

    ax_counts.set_xticks(np.arange(n_arms), [f"Bandit {arm}\n(mean {mean:g})" for arm, mean in enumerate(bandits[0].p)])
    ax_counts.set_yscale("log")
    ax_counts.set_title("Trials given to each ad (log scale)")
    ax_counts.set_ylabel("number of trials")
    ax_counts.legend(loc="upper left")
    ax_best.set_xscale("log")
    ax_best.set_ylim(0, 1.02)
    ax_best.set_title(f"Share of trials so far given to the best ad (Bandit {best_arm})")
    ax_best.set_xlabel("trial (log scale)")
    ax_best.set_ylabel("share of trials")
    ax_best.legend()
    fig.tight_layout()
    return Visualization(output_dir).save(fig, "comparison.png")


def main():
    """
    Run both algorithms on ``BANDIT_REWARD`` for ``NUMBER_OF_TRIALS`` trials,
    store the rewards in ``output/rewards.csv``, log the results and save the plots.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "rewards.csv"

    bandits = [
        EpsilonGreedy(BANDIT_REWARD, epsilon=EPSILON, precision=PRECISION, seed=SEED).experiment(NUMBER_OF_TRIALS),
        ThompsonSampling(BANDIT_REWARD, precision=PRECISION, prior_mean=PRIOR_MEAN,
                         prior_precision=PRIOR_PRECISION, seed=SEED).experiment(NUMBER_OF_TRIALS),
    ]
    for i, bandit in enumerate(bandits):
        bandit.report(csv_path, append=i > 0)

    visualization = Visualization(OUTPUT_DIR)
    for bandit in bandits:
        visualization.plot1(bandit)
    visualization.plot2(bandits)
    comparison(bandits, OUTPUT_DIR)


if __name__=='__main__':
    main()
