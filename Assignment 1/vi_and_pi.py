### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """ policy maps from states to actions. this function evaluates a particular policy and returns the state values for it"""
    
    value_function = np.zeros(nS)
    while True:
        delta = 0
        for state in range(nS):
            v = value_function[state]
            new_value = 0
            for (probability, next_state, reward, last) in P[state][policy[state]]: #policy[state] is action
                new_value += (probability * ( reward + gamma * value_function[next_state]))
            value_function[state] = new_value
            delta = max(delta, abs(new_value - v))
        if delta<tol:
            break
        
    return value_function

def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    
    """ value_from_policy is the values as calculated by the policy evaluation function. it is an array
       policy is an array and is the old policy
       
       
       return a new_policy of form np.ndarray[nS]  where each element is an int
        where each index value corresponds to the optimal action according to the provided value function
    
    """
    
    new_policy = np.zeros(nS, dtype='int')
    
    for state in range(nS):
        best_action, best_q = None, -float("inf")
        for action in range(nA):
            q_new = 0
            for (probability, next_state, reward, last) in P[state][action]:
                q_new += probability * (reward + gamma * value_from_policy[next_state])
            if q_new > best_q:
                best_action, best_q = action, q_new
        new_policy[state] = best_action
        
    return new_policy
  
def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    
    while True:
        flag = True
        value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)
        new_policy = policy_improvement(P, nS, nA, value_function, policy, gamma)
        for i in range(nS):
            if policy[i] != new_policy[i]:
                policy = new_policy
                flag = False
                break
        if flag == True:
            break
    return value_function, policy

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype = int)
    
    while True:
        delta = 0
        for state in range(nS):
            v = value_function[state]
            v_max, best_action = -float("inf"), None
            for action in range(nA):
                v_new = 0
                for (probability, next_state, reward, last) in P[state][action]:
                    v_new += (probability * ( reward + gamma * value_function[next_state]))
                if v_new > v_max:
                    v_max = v_new
                    best_action = action
            policy[state] = best_action
            value_function[state] = v_max                   
            delta = max(delta, abs(v_max - v))
        if delta< tol:
            break
    return value_function, policy
    
def render_single(env, policy, max_steps=100):
  """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

  episode_reward = 0
  ob = env.reset()
  for t in range(max_steps):
    env.render()
    time.sleep(0.55)
    a = policy[ob]
    ob, rew, done, _ = env.step(a)
    episode_reward += rew
    if done:
      break
  env.render();
  if not done:
    print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
  else:
  	print("Episode reward: %f" % episode_reward)
      
if __name__ == "__main__":

	# comment/uncomment these lines to switch between deterministic/stochastic environments
	#env = gym.make("Deterministic-4x4-FrozenLake-v0")
	#env = gym.make("Stochastic-4x4-FrozenLake-v0")
    env = gym.make("Deterministic-8x8-FrozenLake-v0")

    print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)

    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    
    render_single(env, p_pi, 100)

    print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)
    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    
    render_single(env, p_vi, 100)

        
            
        
        
        
          
                
        
    
    