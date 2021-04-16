import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

INIT_STATE = 0
TERMINAL_STATE = 3
# state that reverse the result of the action
REVERSE_STATE = 1
LEFT = -1
RIGHT = 1
ACTIONS = [LEFT, RIGHT]

def generate_an_episode(starting_state, greedy_action, prob_right):
    state = starting_state
    reward = 0
    while state != TERMINAL_STATE:
        if np.random.random() <= prob_right:
            act = RIGHT
        else:
            act = LEFT

        if state == REVERSE_STATE:
            state -= act
        else:
            state += act
        reward -= 1
        state = max(state, 0)
    return reward

epsilon = 0.1
# epsilon-greedy left
prob_right = epsilon / 2

rewards = []
for i in range(2000):
    rewards.append(generate_an_episode(INIT_STATE, RIGHT, prob_right))
print(np.mean(rewards))

# epsilon-greedy right
prob_right = (1 - epsilon) + epsilon / 2

rewards = []
for i in range(2000):
    rewards.append(generate_an_episode(INIT_STATE, RIGHT, prob_right))
print(np.mean(rewards))

# probabilities of going right
probs = np.sort(np.arange(0.02, 1, 0.02).tolist() + [2/3])

avg_rewards = []
std_rewards = []
for k, p in enumerate(probs):
    print(k, end=',')
    rewards = []
    for i in range(2000):
        rewards.append(generate_an_episode(INIT_STATE, RIGHT, p))
    avg_rewards.append(np.mean(rewards))
    std_rewards.append(np.std(rewards))

plt.figure()
plt.errorbar(probs, avg_rewards, yerr=std_rewards)
idx = np.where(probs == 2/3)[0][0]
plt.scatter([probs[idx]], avg_rewards[idx], label='optimal probability')
plt.legend(loc='lower right')
plt.xlabel('Probability of moving right')
plt.ylabel('Value of state S')
plt.show()