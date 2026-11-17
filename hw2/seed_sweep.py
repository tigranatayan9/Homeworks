"""
Evidence for the bonus section of the README: run each algorithm setting over
many seeds and summarise the final cumulative regret, since a single run can be
very misleading. Takes about two minutes.

Run it with ``python seed_sweep.py``.
"""
import numpy as np
from loguru import logger

from Bandit import BANDIT_REWARD, NUMBER_OF_TRIALS, EpsilonGreedy, ThompsonSampling

N_SEEDS = 200
STUCK_REGRET = 5000

SETTINGS = {
    "EpsilonGreedy, epsilon = 1/t": lambda seed: EpsilonGreedy(BANDIT_REWARD, epsilon=1, seed=seed),
    "EpsilonGreedy, epsilon = 2/t": lambda seed: EpsilonGreedy(BANDIT_REWARD, epsilon=2, seed=seed),
    "EpsilonGreedy, epsilon = 5/t": lambda seed: EpsilonGreedy(BANDIT_REWARD, epsilon=5, seed=seed),
    "EpsilonGreedy, epsilon = 10/t": lambda seed: EpsilonGreedy(BANDIT_REWARD, epsilon=10, seed=seed),
    "ThompsonSampling, prior precision 1": lambda seed: ThompsonSampling(BANDIT_REWARD, prior_precision=1, seed=seed),
    "ThompsonSampling, prior precision 0.01": lambda seed: ThompsonSampling(BANDIT_REWARD, prior_precision=0.01, seed=seed),
}


def final_regrets(make_bandit, n_seeds=N_SEEDS, n_trials=NUMBER_OF_TRIALS):
    """
    Run one setting once per seed and collect the final cumulative regrets.

    :param make_bandit: builds a fresh algorithm from a seed
    :type make_bandit: callable
    :param n_seeds: number of seeds (0 .. n_seeds - 1)
    :type n_seeds: int
    :param n_trials: number of trials per run
    :type n_trials: int
    :return: final cumulative regret of every run
    :rtype: numpy.ndarray
    """
    return np.array([make_bandit(seed).experiment(n_trials).cumulative_regrets()[-1] for seed in range(n_seeds)])


if __name__ == "__main__":
    logger.disable("Bandit")
    logger.info(f"Final cumulative regret over {N_SEEDS} seeds, {NUMBER_OF_TRIALS:,} trials each")
    for label, make_bandit in SETTINGS.items():
        regrets = final_regrets(make_bandit)
        logger.info(f"{label:<40} mean {regrets.mean():8.1f} | median {np.median(regrets):6.1f} | "
                    f"90th pct {np.percentile(regrets, 90):8.1f} | max {regrets.max():8.1f} | "
                    f"stuck (> {STUCK_REGRET}) {np.mean(regrets > STUCK_REGRET):6.1%}")
