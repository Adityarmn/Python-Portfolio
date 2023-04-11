import argparse
import os
from pickle import TRUE
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from env import SumoEnvironmentPZ
from dqn_marl import Memory
from dqn_marl import DQNAgent
import random
if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    prs.add_argument("-route", dest="route", type=str, default='net_jkt-new/jkt-new.rou.xml', help="Route definition xml file.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.0001, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.95, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.01, required=False, help="Epsilon.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=5, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=30, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=82800, required=False, help="The time in seconds the simulation must end..\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    args = prs.parse_args()

    out_csv = 'output/onlinelearning.csv'
    out_csv2 = 'output/simulationvalue.csv'

    #nyoba di single intersection
    #cfg_file_='net-single/single.sumocfg'
    #net_file_= 'net-single/single.net.xml'

    cfg_file_='net_jkt-new/jkt-new.sumocfg'
    net_file_ = 'net_jkt-new/jkt-new.net.xml'
    runs = 1 #mode testing run cukup sekali namun waktu simulasinya panjang.
    batch_size = 256

    env = SumoEnvironmentPZ(net_file = net_file_,
                          #single_agent=True,
                          cfg_file=cfg_file_,
                          route_file=args.route,
                          out_csv_name=out_csv,
                          out_csv_name2 = out_csv2,
                          use_gui=False,
                          num_seconds=args.seconds,
                          min_green=args.min_green,
                          max_green=args.max_green)

    for run in range(1, runs+1):
        env.reset()
        state = {ts: env.observe(ts) for ts in env.agents}
        dqn_agent = {ts: DQNAgent(state_size=env.observation_space(ts).shape[0],action_size=env.action_space(ts).n) for ts in env.agents}
        memories = {ts: Memory(10000) for ts in env.agents}
        done = False
        reward_all = 0

        for agent in dqn_agent:
            agent_model = 'output-agent/chosen/chosen-best-reward/'+str(agent)+'.h5'#isi sesuai agent terakhir atau pilih dengan reward terbaik.
            dqn_agent[agent].load(agent_model)

        for agent in env.agent_iter():
            #load agent di dalam loop jika agent akan selalu di reset ke saved model?

            states_next, reward, done, info = env.last()
            action = dqn_agent[agent].act(state[agent]) if not done else None
            env.step(action)

            memories[agent].remember(state=state[agent], action=action, reward=reward, state_next=states_next, done=done)
            size = memories[agent].pointer
            batch = random.sample(range(size), size) if size < batch_size else random.sample(range(size), batch_size)

            if len(memories[agent].data) > batch_size:
                dqn_agent[agent].replay(*memories[agent].sample(batch))

            if done :
                dqn_agent[agent].save('output-agent/testing/05-07/'+str(agent)+'.h5')

            state[agent] = states_next
            reward_all += reward


        print("Score: {s}, Goal: {g}".format(s=reward_all, g=done))
        env.unwrapped.env.save_csv(out_csv, run)
        env.close()
