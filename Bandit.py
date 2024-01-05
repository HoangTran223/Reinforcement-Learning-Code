import numpy as np
import random
class BernoulliBandit:
    
    def __init__(self, p, verbose=True):
        self.p = p
        if verbose:
            print("Creating BernoulliBandit with p = {:.2f}".format(p))
    
    def pull(self):
        return np.random.binomial(10, self.p) 

class BanditsGame:
    
    def __init__(self, K, T, verbose=True):
        
        self.T = T
        self.K = K
        self.bandits = [BernoulliBandit(np.random.uniform(), verbose) for i in range(K)]
        self.verbose = verbose

    def run_stochastic(self):
        
        results = np.zeros((self.T))
        
        for t in range(self.T):
            k = random.randrange(self.K)
            results[t] = self.bandits[k].pull()
            if self.verbose:
                print("T={} \t Playing bandit {} \t Reward is {:.2f}".format(t, k, results[t]))
        
        return results

game = BanditsGame(K=3, T=20)
game.run_stochastic()

def run_simulation(n_runs, runs_per_game, K, T):
    
    results = np.zeros((K,T))
    
    for run in range(n_runs):

        run_results = np.zeros((K,T))

        for run in range(runs_per_game):
            game = BanditsGame(K=K, T=T, verbose=False)
            run_results += game.run_stochastic()

        results += run_results / runs_per_game
    
    results = results / n_runs
    
    return results


stochastic_results = run_simulation(n_runs=10, runs_per_game=100, K=3, T=1000)
stochastic_results = stochastic_results.mean(axis=0)
print("Mean reward: {:.2f}".format(stochastic_results.mean()))
print("G: {:.2f}".format(stochastic_results.sum()))