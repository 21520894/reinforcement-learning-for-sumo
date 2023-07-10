from sumo_rl import SumoEnvironment
from QLAgent import QLAgent
import argparse
from sumo_rl import exploration
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy

parser = argparse.ArgumentParser()
parser.add_argument("--net")
parser.add_argument("--route")
parser.add_argument("--output", default= './')
parser.add_argument("--train_time", default = 100000)
parser.add_argument("--play_time",default = 50000)
args = parser.parse_args()
print(args)


net = args.net
route = args.route
train_time = int(args.train_time)
play_time = int(args.play_time)
out_csv = args.output


## HYPERPARAMETERS
min_green = 5
max_green = 30
alpha = 0.1
gamma = 0.99

init_epsilon = 0.05
min_epsilon = 0.005
decay = 1



from sumo_rl import SumoEnvironment
def qlearn(net,route,seconds = 1000):
    env = SumoEnvironment(
        net_file=net,
        route_file=route,
        out_csv_name = out_csv,
        use_gui=False,
        num_seconds=seconds,
        min_green=min_green,
        max_green=max_green,
    )

    initial_states = env.reset()
    # print(initial_states)
    ql_agents = {
        ts: QLAgent(
            starting_state=env.encode(initial_states[ts], ts),
            state_space=env.observation_space,
            action_space=env.action_space,
            alpha=alpha,
            gamma=gamma,
            exploration_strategy=EpsilonGreedy(
                initial_epsilon=init_epsilon, min_epsilon=min_epsilon, decay=decay
            ),
        )
        for ts in env.ts_ids
    }

    done = {"__all__": False}
    while not done["__all__"]:
        actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}
        # print(actions)
        # break
        # print("\n")
        s, r, done, _ = env.step(action=actions)

        for agent_id in ql_agents.keys():
            ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])
    env.save_csv(out_csv, '_train')
    env.close()

    return ql_agents

def Play(net, rou , ql_agents, seconds = 1000, gui = False, out ='./'):
  env = SumoEnvironment(
        net_file=net,
        route_file=route,
        out_csv_name = out_csv,
        use_gui=False,
        num_seconds=seconds,
        min_green=min_green,
        max_green=max_green,
    )
  s = env.reset()
  # print(s)
  done = {"__all__": False}
  infos = []
  while not done["__all__"]:
      # actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}
      actions = {ts: ql_agents[ts].bestAction(env.encode(s[ts],ts)) for ts in ql_agents.keys()}
      # print(actions)
      s, r, done, _ = env.step(action=actions)
      # print(s)
      # print(r)
  env.save_csv(out_csv,'_play')
if __name__ == '__main__':
  agent = qlearn(net,route,seconds = train_time)
  if (play_time > 0):
    Play(net,route,agent,seconds = play_time)