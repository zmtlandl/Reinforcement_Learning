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
        pos = self.discretise(observation[0], mini=-1, maxi=1)
        vel = self.discretise(observation[1], mini=-1, maxi=1)
        angle = self.discretise(observation[2], mini=-1, maxi=1)
        angle_vel = self.discretise(observation[3], mini=-1, maxi=1)
        return pos* nval + vel * nval* nval + angle* nval + angle_vel * nval * nval

    '''
        Monte Carlo algorithms 
    '''
    def QLearning(self, alpha, gam, epl):
        Q = {}
        output = []
        n = 0
        for i in range(self.episodes):
            n += 1
            print("episodes:",n)
            done = False
            observation = self.env.reset()  # reset all variables to the initial state
            s = self.observation_vers_etat(observation)
            if (s, 0) not in Q:
                Q[(s, 0)], Q[(s, 1)] = np.random.rand(1)[0], 1 - np.random.rand(1)[0]
            alt = np.random.rand(1)[0]
            best_a = 0 if Q[(s, 0)] > Q[(s, 1)] else 1
            a = best_a if alt > epl else np.random.randint(2)
            nbInt = 0
            while not done:
                # play (s,a)
                observation, reward, done, info = self.env.step(a)
                next_s = self.observation_vers_etat(observation)
                if (next_s, 0) not in Q:
                    Q[(next_s, 0)], Q[(next_s, 1)] = np.random.rand(1)[0], 1 - np.random.rand(1)[0]
                alt = np.random.rand(1)[0]
                best_a = 0 if Q[(next_s, 0)] > Q[(next_s, 1)] else 1
                next_a = best_a if alt > epl else np.random.randint(2)
                Q[(s,a)] += alpha * (reward + gam * Q[(next_s, best_a)] - Q[(s,a)])
                s = next_s
                a = next_a
                nbInt += 1
                self.env.render()
            output.append(nbInt)
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



nval = 20
N = nval ** 4
episodes = 5000
gam = 1
alpha = 0.1
epl = 0.1
cartPole = cartPole(nval,episodes)

output = cartPole.QLearning(alpha,gam,epl)
cartPole.plotAverageOfIterationsPerEpisode(output)
cartPole.env.close()

























