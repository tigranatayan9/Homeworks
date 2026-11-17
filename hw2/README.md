# Homework 2 — A/B testing with multi-armed bandits

Four advertisements (bandits) with mean rewards `[1, 2, 3, 4]` are tested for 20,000 trials each with
**Epsilon-Greedy** (epsilon decaying as 1/t) and **Thompson Sampling** (Gaussian rewards, known precision).

## How to run

```bash
pip install -r requirements.txt
python Bandit.py          # the experiment: plots, CSV and logged results, about 5 s
python seed_sweep.py      # optional, evidence for the bonus section, about 2 min
```

Everything `Bandit.py` produces goes to `output/`:

| File | Contents |
|---|---|
| `rewards.csv` | one row per trial: `Bandit` (0-based ad index), `Reward`, `Algorithm` |
| `plot1_EpsilonGreedy.png`, `plot1_ThompsonSampling.png` | learning process: running average reward and each ad's estimated mean, linear and log scale |
| `plot2_cumulative_reward_and_regret.png` | cumulative reward and cumulative regret of both algorithms |
| `comparison.png` | trials given to each ad, and the share of trials given to the best ad over time |

Cumulative reward and cumulative regret (total and per trial) are logged with `loguru`.

## Design

- The `Bandit` abstract class is kept intact. I only added docstrings and helpers shared by both algorithms
  (reward drawing, the trial loop, regret, CSV writing).
- An `EpsilonGreedy` or `ThompsonSampling` object is one algorithm playing all four ads. Its `p` is the list
  of true mean rewards. `pull()` chooses an ad and observes its reward, `update(arm, reward)` learns from it,
  `experiment(n_trials)` runs the trials, and `report(csv_path, append)` writes the CSV and logs the results.
- Ad *i* pays a reward drawn from N(p<sub>i</sub>, 1/τ), with known precision τ = 1.
- **Regret** of a trial = 4 − (true mean of the ad shown). It measures the cost of the choices only, not the
  reward noise.

### Parameter choices

| Parameter | Value | Why |
|---|---|---|
| epsilon | ε<sub>t</sub> = min(1, 5/t) | Plain 1/t explores too little; see the bonus table. 5/t gave the lowest mean regret of the schedules tried. |
| precision τ | 1 | Reward standard deviation 1, the same size as the gap between neighbouring ads. |
| Thompson prior | N(0, 1/0.01) | A vague prior. With N(0, 1), an ad tried once is pulled halfway to 0 and can look bad for hundreds of trials. |
| seed | 42 | Reproducible runs. |

## Results (seed 42)

| | Epsilon-Greedy (5/t) | Thompson Sampling |
|---|---:|---:|
| Cumulative reward | 79,964.69 | 79,937.44 |
| Cumulative regret | 86.00 | 37.00 |
| Trials on the best ad | 99.81% | 99.88% |

Thompson Sampling has less than half the regret. It explores the runner-up ad (mean 3), where it is
actually uncertain, while Epsilon-Greedy explores all ads equally and spends 18 trials on the worst one.
Thompson Sampling's *realized* reward is slightly lower in this run anyway. That gap is reward noise: the
expected rewards are 80,000 − regret, so 79,914 for Epsilon-Greedy and 79,963 for Thompson Sampling.

## Bonus: a better implementation plan

These changes are in priority order. Numbers come from `seed_sweep.py`: 200 seeds, 20,000 trials each. A run
counts as "stuck" when its regret is above 5,000, meaning a worse ad got most of the trials.

| Setting | Mean regret | Median | 90th pct | Max | Runs stuck on a worse ad |
|---|---:|---:|---:|---:|---:|
| Epsilon-Greedy, ε = 1/t | 2,984.8 | 25.0 | 20,005.1 | 40,455.0 | 16.0% |
| Epsilon-Greedy, ε = 2/t | 493.0 | 33.0 | 136.6 | 20,034.0 | 2.0% |
| **Epsilon-Greedy, ε = 5/t** | **75.4** | 70.0 | 94.1 | 484.0 | 0.0% |
| Epsilon-Greedy, ε = 10/t | 129.6 | 128.5 | 154.0 | 211.0 | 0.0% |
| Thompson Sampling, prior N(0, 1) | 321.6 | 29.0 | 977.9 | 3,863.0 | 0.0% |
| **Thompson Sampling, prior N(0, 1/0.01)** | **31.2** | 31.0 | 43.0 | 52.0 | 0.0% |

1. **Judge algorithms over many runs, not one.** A single seed is a noisy sample. With ε = 1/t, seed 42
   happened to stay on the mean-3 ad for about 6,000 trials, giving regret 6,181, while the median over 200
   seeds is 25. Reporting the mean, median and tail of regret over many seeds, and plotting the average regret
   curve with a band, gives a fair verdict. `seed_sweep.py` is a first step.
2. **Scale the epsilon schedule to the problem.** Plain ε = 1/t explores only about ln(20,000) ≈ 10 times in
   the whole run, so one unlucky early reward can lock in a worse ad for good (16% of runs). The schedule
   ε<sub>t</sub> = min(1, cK/(d²t)) from Auer, Cesa-Bianchi & Fischer (2002), with K ads and d the smallest
   gap between ad means, keeps the 1/t decay but explores enough. Here cK/d² = 5 removed every stuck run.
   Showing every ad once before exploiting also removes the dependence on `argmax` breaking ties towards ad 0.
3. **Use a vague or data-driven prior in Thompson Sampling.** A tight prior far from the true means (N(0, 1)
   here) shrinks rarely-tried ads towards 0 and gives a heavy tail: mean regret 321.6 against 31.2 for a vague
   prior. In production, set the prior from historical performance of similar ads.
4. **Match the reward model to real ad data.** The precision of real rewards is rarely known. A Normal–Gamma
   prior learns the mean and the precision together. When the reward is a click or no click, Beta–Bernoulli
   Thompson Sampling is the natural model.
5. **Compare algorithms on the same random numbers.** Separate the environment (ads and their rewards) from
   the agents (policies), and let every agent read rewards from the same pre-drawn table. Differences then come
   from the policies, not from noise. This also avoids the result above where lower regret came with lower
   realized reward.
6. **Add UCB1 as a third baseline.** It is deterministic, needs no tuning parameter, and has a logarithmic
   regret guarantee, so it is a useful reference point between the two.
