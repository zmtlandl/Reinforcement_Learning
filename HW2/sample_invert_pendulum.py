#!/usr/bin/env python

import random
import pandas as pd
import numpy as np

import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt


class AgentLearning(object):
    '''' Agent that can learn via Q-learning. '''
    def __init__(self, env, alpha=0.1, epsilon=1.0, gamma=0.9):
        self.env = env
        self.alpha = alpha          # Learning factor
        self.epsilon = epsilon
        self.gamma = gamma          # Discount factor
        self.Q_table = dict()
        self._set_seed()
        # Following variables for statistics
        self.training_trials = 0
        self.testing_trials = 0

    def _set_seed(self):
        ''' Set random seeds for reproducibility. '''
        np.random.seed(21)
        random.seed(21)

    def build_state(self, features):
        ''' Build state by concatenating features (bins) into 4 digit int. '''
        return int("".join(map(lambda feature: str(int(feature)), features)))

    def create_state(self, obs):
        ''' Create state variable from observation.

        Args:
            obs: Observation list with format [horizontal position, velocity,
                 angle of pole, angular velocity].
        Returns:
            state: State tuple
        '''
        cart_position_bins = pd.cut([-2.4, 2.4], bins=10, retbins=True)[1][1:-1]
        pole_angle_bins = pd.cut([-2, 2], bins=10, retbins=True)[1][1:-1]
        cart_velocity_bins = pd.cut([-1, 1], bins=10, retbins=True)[1][1:-1]
        angle_rate_bins = pd.cut([-3.5, 3.5], bins=10, retbins=True)[1][1:-1]
        state = self.build_state([np.digitize(x=[obs[0]], bins=cart_position_bins)[0],
                                 np.digitize(x=[obs[1]], bins=pole_angle_bins)[0],
                                 np.digitize(x=[obs[2]], bins=cart_velocity_bins)[0],
                                 np.digitize(x=[obs[3]], bins=angle_rate_bins)[0]])
        return state

    def choose_action(self, state):
        ''' Given a state, choose an action.

        Args:
            state: State of the agent.
        Returns:
            action: Action that agent will take.
        '''
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            # Find max Q value
            max_Q = self.get_maxQ(state)
            actions = []
            for key, value in self.Q_table[state].items():
                if value == max_Q:
                    actions.append(key)
            if len(actions) != 0:
                action = random.choice(actions)
        return action

    def create_Q(self, state, valid_actions):
        ''' Update the Q table given a new state/action pair.

        Args:
            state: List of state booleans.
            valid_actions: List of valid actions for environment.
        '''
        if state not in self.Q_table:
            self.Q_table[state] = dict()
            for action in valid_actions:
                self.Q_table[state][action] = 0.0
        return

    def get_maxQ(self, state):
        ''' Find the maximum Q value in a given Q table.

        Args:
            Q_table: Q table dictionary.
            state: List of state booleans.
        Returns:
            maxQ: Maximum Q value for a given state.
        '''
        maxQ = max(self.Q_table[state].values())
        return maxQ

    def learn(self, state, action, prev_reward, prev_state, prev_action):
        ''' Update the Q-values

        Args:
            state: State at current time step.
            action: Action at current time step.
            prev_reward: Reward at previous time step.
            prev_state: State at previous time step.
            prev_action: Action at previous time step.
        '''
        # Updating previous state/action pair so I can use the 'future state'
        self.Q_table[prev_state][prev_action] = (1 - self.alpha) * \
            self.Q_table[prev_state][prev_action] + self.alpha * \
            (prev_reward + (self.gamma * self.get_maxQ(state)))
        return


def moving_average(data, window_size=10):
    ''' Calculate moving average with given window size.
    Args:
        data: List of floats.
        window_size: Integer size of moving window.
    Returns:
        List of rolling averages with given window size.
    '''
    sum_vec = np.cumsum(np.insert(data, 0, 0))
    moving_ave = (sum_vec[window_size:] - sum_vec[:-window_size]) / window_size
    return moving_ave


def display_stats(agent, training_totals, testing_totals, history):
    ''' Print and plot the various statistics from q-learning data.
    Args:
        agent: Agent containing variables useful for post analysis
        training_totals: List of training rewards per episode.
        testing_totals: List of testing rewards per episode.
        epsilon_hist: List of all epsilon values.
    '''
    print('******* Training Stats *********')
    print('Average: {}'.format(np.mean(training_totals)))
    print('Standard Deviation: {}'.format(np.std(training_totals)))
    print('Minimum: {}'.format(np.min(training_totals)))
    print('Maximum: {}'.format(np.max(training_totals)))
    print('Number of training episodes: {}'.format(agent.training_trials))
    print()
    print('******* Testing Stats *********')
    print('Average: {}'.format(np.mean(testing_totals)))
    print('Standard Deviation: {}'.format(np.std(testing_totals)))
    print('Minimum: {}'.format(np.min(testing_totals)))
    print('Maximum: {}'.format(np.max(testing_totals)))
    print('Number of testing episodes: {}'.format(agent.testing_trials))
    fig = plt.figure(figsize=(10, 7))
    # Plot Parameters plot
    ax1 = fig.add_subplot(311)
    ax1.plot([num + 1 for num in range(agent.training_trials)],
             history['epsilon'],  # epsilon_hist,
             color='b',
             label='Exploration Factor (Epsilon)')
    ax1.plot([num + 1 for num in range(agent.training_trials)],
             history['alpha'],  #alpha_hist,
             color='r',
             label='Learning Factor (Alpha)')
    ax1.set(title='Paramaters Plot',
            ylabel='Parameter values',
            xlabel='Trials')

    # Plot rewards
    ax2 = fig.add_subplot(312)
    ax2.plot([num + 1 for num in range(agent.training_trials)],
             training_totals,
             color='m',
             label='Training',
             alpha=0.4, linewidth=2.0)
    total_trials = agent.training_trials + agent.testing_trials
    ax2.plot([num + 1 for num in range(agent.training_trials, total_trials)],
             testing_totals,
             color='k',
             label='Testing', linewidth=2.0)
    ax2.set(title='Reward per trial',
            ylabel='Rewards',
            xlabel='Trials')

    # Plot rolling average rewards
    ax3 = fig.add_subplot(313)
    window_size = 10
    train_ma = moving_average(training_totals, window_size=window_size)
    train_epi = [num+1 for num in range(agent.training_trials-(window_size-1))]
    ax3.plot(train_epi, train_ma,
             color='m',
             label='Training',
             alpha=0.4, linewidth=2.0)
    test_ma = moving_average(testing_totals, window_size=window_size)
    total_trials = total_trials - (window_size*2) + 2
    test_epi = [num+1 for num in range(agent.training_trials-(window_size-1), total_trials)]
    ax3.plot(test_epi, test_ma,
             color='k',
             label='Testing', linewidth=2.0)
    ax3.set(title='Rolling Average Rewards',
            ylabel='Reward',
            xlabel='Trials')

    fig.subplots_adjust(hspace=0.5)
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')


def save_info(agent, training_totals, testing_totals):
    ''' Write statistics into text file.
    Args:
        agent: Agent containing variables useful for post analysis
        training_totals: List of training rewards per episode.
        testing_totals: List of testing rewards per episode.
    '''
    with open('CartPole-v0_stats.txt', 'w') as file_obj:
        file_obj.write('/-------- Q-Learning --------\\\n')
        file_obj.write('\n/---- Training Stats ----\\\n')
        file_obj.write('Average: {}\n'.format(np.mean(training_totals)))
        file_obj.write('Standard Deviation: {}\n'.format(np.std(training_totals)))
        file_obj.write('Minimum: {}\n'.format(np.min(training_totals)))
        file_obj.write('Maximum: {}\n'.format(np.max(training_totals)))
        file_obj.write('Number of training episodes: {}\n'.format(agent.training_trials))
        file_obj.write('\n/---- Testing Stats ----\\\n')
        file_obj.write('Average: {}\n'.format(np.mean(testing_totals)))
        file_obj.write('Standard Deviation: {}\n'.format(np.std(testing_totals)))
        file_obj.write('Minimum: {}\n'.format(np.min(testing_totals)))
        file_obj.write('Maximum: {}\n'.format(np.max(testing_totals)))
        file_obj.write('Number of testing episodes: {}\n'.format(agent.testing_trials))
        file_obj.write('\n/---- Q-Table ----\\')
        for state in agent.Q_table:
            file_obj.write('\n State: ' + str(state) +
                           '\n\tAction: ' + str(agent.Q_table[state]))
    # Save figure and display plot
    plt.savefig('plots.png')
    plt.show()


def q_learning(env, agent):
    '''
    Implement Q-learning policy.

    Args:
        env: Gym enviroment object.
        agent: Learning agent.
    Returns:
        Rewards for training/testing and epsilon/alpha value history.
    '''
    # Start out with Q-table set to zero.
    # Agent initially doesn't know how many states there are...
    # so if a new state is found, then add a new column/row to Q-table
    valid_actions = [0, 1]
    tolerance = 0.001
    training = True
    training_totals = []
    testing_totals = []
    history = {'epsilon': [], 'alpha': []}
    for episode in range(800):  # 688 testing trials
        episode_rewards = 0
        obs = env.reset()
        # If epsilon is less than tolerance, testing begins
        if agent.epsilon < tolerance:
            agent.alpha = 0
            agent.epsilon = 0
            training = False
        # Decay epsilon as training goes on
        agent.epsilon = agent.epsilon * 0.99  # 99% of epsilon value
        for step in range(200):        # 200 steps max
            state = agent.create_state(obs)           # Get state
            agent.create_Q(state, valid_actions)      # Create state in Q_table
            action = agent.choose_action(state)         # Choose action
            obs, reward, done, info = env.step(action)  # Do action
            episode_rewards += reward                   # Receive reward
            # Skip learning for first step
            if step != 0:
                # Update Q-table
                agent.learn(state, action, prev_reward, prev_state, prev_action)
            prev_state = state
            prev_action = action
            prev_reward = reward
            if done:
                # Terminal state reached, reset environment
                break
        if training:
            training_totals.append(episode_rewards)
            agent.training_trials += 1
            history['epsilon'].append(agent.epsilon)
            history['alpha'].append(agent.alpha)
        else:
            testing_totals.append(episode_rewards)
            agent.testing_trials += 1
            # After 100 testing trials, break. Because of OpenAI's rules for solving env
            if agent.testing_trials == 100:
                break
    return training_totals, testing_totals, history


def main():
    ''' Execute main program. '''
    # Create a cartpole environment
    # Observation: [horizontal pos, velocity, angle of pole, angular velocity]
    # Rewards: +1 at every step. i.e. goal is to stay alive
    env = gym.make('CartPole-v0')
    # Set environment seed
    env.seed(21)
    
    agent = AgentLearning(env, alpha=0.9, epsilon=1.0, gamma=0.9)
    training_totals, testing_totals, history = q_learning(env, agent)
    display_stats(agent, training_totals, testing_totals, history)
    save_info(agent, training_totals, testing_totals)
    # Check if environment is solved
    if np.mean(testing_totals) >= 195.0:
        print("Environment SOLVED!!!")
    else:
        print("Environment not solved.",
                "Must get average reward of 195.0 or",
                "greater for 100 consecutive trials.")
    


if __name__ == '__main__':
    ''' Run main program. '''
    main()