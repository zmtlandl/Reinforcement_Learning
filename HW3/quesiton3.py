import numpy as np
import gym
import matplotlib.pyplot as plt


class cartPole():
    def __init__(self, nval,  episodes):
        self.nval = nval # number of values for each observation variable
        self.N = self.nval ** 4
        self.episodes = episodes #total number of episodes to play
        self.A = (0,1)  # actions
        self.env = gym.make('CartPole-v1')


    def discretise(self, x, mini, maxi):
        # discretise x
        # return an integer between 0 and nval - 1
        if x < mini: x = mini
        if x > maxi: x = maxi
        return int(np.floor((x - mini) * nval / (maxi - mini + 0.0001)))

    def observation_vers_etat(self, observation):
        angle = self.discretise(observation[0], mini=-1, maxi=1)
        angle_vel = self.discretise(observation[1], mini=-1, maxi=1)
        angle_acc = self.discretise(observation[2], mini=-1, maxi=1)
        return np.sin(angle) - angle_vel * nval - angle_acc * nval * nval

    '''
        Monte Carlo algorithms 
    '''

    def MonteCarlo_OnPolicy_FirstVisit(self,gam, epl):
        """
        Monte carlo algorithm with On-policy - First visit
        :return: the average number of iterations per episode
        """
        '''
            initialize Occ(s,a) : how many times we visited (s,a) (first iteration)
            initialize Q(s,a): the value function for (s,a)
            initialize Acc(s,a): reward function
            initialize p(s,a): politic 
            initialize episode : visited states 

        '''
        Occ, Q, Acc, p = {}, {}, {}, {}
        output = []
        n = 0
        for i in range(self.episodes):
            '''
                play an episode 
            '''
            episode = []
            nbIt = 0
            ep = epl
            done = False
            observation = self.env.reset()  # reset all variables to the initial state
            n += 1
            print("episodes:",n)
            while not done:
                ep = self.EpsilonDecreazing(ep)
                s = self.observation_vers_etat(observation)
                if (s, 0) not in p:
                    p[(s, 0)], p[(s, 1)] = np.random.rand(1)[0], 1 - np.random.rand(1)[0]
                alt = np.random.rand(1)[0]
                a = 0 if alt <= p[(s, 0)] else 1
                observation, reward, done, info = self.env.step(a)
                episode.append((s, a))
                self.env.render()
                nbIt += 1
            output.append(nbIt)
            state_visited = set()
            for t in range(len(episode)):
                item = episode[t]
                if item not in state_visited:
                    state_visited.add(item)
                    G = (1 - gam ** (nbIt - t)) / (1 - gam)
                    Acc[item] = G if item not in Acc else Acc[item] + G
                    Occ[item] = 1 if item not in Occ else Occ[item] + 1
                    Q[item] = Acc[item] / Occ[item]
            for state, action in state_visited:
                best_a_state = 0 if ((state, 1) not in Q or ((state, 0) in Q and Q[(state, 0)] > Q[(state, 1)])) else 1
                for action in self.A:
                    p[(state, action)] = 1 - ep + ep / 2 if best_a_state == action else ep / 2
        return output

    def plotAverageOfIterationsPerEpisode(self,results):
        episodes = np.arange(0,len(results),100)
        y = []
        sum, count = 0, 0
        for r in results:
            count += 1
            sum += r
            if count == 100 :
                y.append(sum/100)
                sum, count = 0, 0
        fig, ax = plt.subplots()
        print("episodes:",episodes,"y:",y)
        ax.plot(episodes, y)
        ax.set(xlabel='episodes', ylabel='average iterations',
               title='cart pole')
        ax.grid()
        fig.savefig("test.png")
        plt.show()


    def EpsilonDecreazing(self, epl):
        '''
         This is a decreazing function for calculating epsilon parameter
        :param epl:
        :return:
        '''
        epl = - epl / 2 + 1 / 2
        return epl

    def gamVariation(self, min, max):
        plots = []
        gam = min
        while gam <= max:
            plots.append(cartPole.MonteCarlo_OnPolicy_FirstVisit(epl, gam))
            gam += 0.2
        self.CompareAlgorithms(plots)



nval = 20
N = nval ** 4
episodes = 5000
gam = 1
alpha = 0.1
epl = 0.1
cartPole = cartPole(nval,episodes)
cartPole.gamVariation(1,1)

output_1 = cartPole.MonteCarlo_OnPolicy_FirstVisit(epl,gam)

cartPole.plotAverageOfIterationsPerEpisode(output_1)
cartPole.env.close()

























