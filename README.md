# Tuning TD3 for LunarLanderContinuous-v2

## Description
This project aims to compare two implementations of the TD3 algorithm (Twin Delayed Deep Deterministic Policy Gradient) applied to the `LunarLanderContinuous-v2` environment. We developed our own TD3 implementation and compared it with the implementation provided by the `Stable Baselines3` library.

The performance of both implementations is evaluated based on several metrics:
- Actor loss
- Critic loss
- Average rewards

## Methodology
1. **Define and implement the TD3 algorithm** with standard hyperparameters.
2. **Train and test the algorithm** on the `LunarLanderContinuous-v2` environment.
3. **Compare results** with those obtained using `Stable Baselines3`.
4. **Analyze performance** through multiple runs and hyperparameter settings to observe learning trends.

## TD3 Algorithm
The TD3 algorithm is an enhanced version of DDPG, designed to improve stability and efficiency in learning. It uses two independent critic networks to estimate state-action values and reduce overestimation. Here are some of the key hyperparameters used:
- Batch size: `128`
- Actor learning rate: `1e-3`
- Critic learning rate: `1e-3`
- Soft target update rate: `0.005`
- Discount factor: `0.99`
- Replay buffer size: `1 million`

## Comparison with Stable Baselines3
We compared our implementation with the `Stable Baselines3` implementation using the same set of hyperparameters. Both implementations were tested on 5 different seeds to ensure robustness and consistency in the results.

## Results
We evaluated the following performance indicators:
- **Actor and critic losses:** Tracking how the loss changes over time during training.
- **Average rewards:** Measuring the cumulative rewards obtained during the learning process.

While both implementations showed similar trends in terms of actor and critic loss reduction, our implementation exhibited less fluctuation in critic loss, suggesting a potentially more stable learning process.

## Statistical Tests
We also performed statistical analysis, using Welch's t-test, to rigorously compare the performance of the two implementations. The analysis showed that while there are minor differences, the performance between the two implementations is comparable, with no statistically significant advantage for either.

## Conclusion
This project successfully compared two TD3 implementations. Both implementations performed similarly, though further experiments with more seeds and extended hyperparameter tuning could reveal more insights into the subtle differences in their performance.

