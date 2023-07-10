from numpy.core.fromnumeric import choose
import numpy as np
from sumo_rl import exploration
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy

class QLAgent:
    """Q-learning Agent class."""

    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        """Initialize Q-learning agent."""
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {self.state: np.zeros(action_space.n)}
        self.exploration = exploration_strategy
        self.acc_reward = 0

    def act(self):
        """Choose action based on Q-table."""
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        return self.action

    def learn(self, next_state, reward, done=False):
        """Update Q-table with new experience."""
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_space.n)

        s = self.state
        s1 = next_state
        a = self.action
        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
            reward + self.gamma * max(self.q_table[s1]) - self.q_table[s][a]
        )
        self.state = s1
        self.acc_reward += reward

    def q_table(self):
      return self.q_table

    def bestAction(self,state,random = False):
      action = self.action_space.sample()
      if state not in self.q_table:
        if (random == False ): return int(action)
        print("NEVER MET THIS STATE BEFORE")
        # return int(self.action_space.sample())
        max_dist = float('inf')
        for learned_state in self.q_table:
          dist = np.linalg.norm(np.array(state) - np.array(learned_state))
          if dist < max_dist:
            max_dist = dist
            action = self.q_table[learned_state].argmax()
      else:
        action = self.q_table[state].argmax()
      return int(action)