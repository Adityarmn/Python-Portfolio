import argparse
import os
import sys
from datetime import datetime

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from collections import deque
from env_baru import SumoEnvironmentPZ
from env_baru import SumoEnvironment
from boots import DQNAgentBoostrapped
from boots import Memory
from boots import MemoryHeads
import numpy as np
import random
from queue import Queue
from copy import deepcopy
import collections
import copy
if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #prs.add_argument("-route", dest="route", type=str, default='nets/single-intersection.rou.xml', help="Route definition xml file.\n")
    prs.add_argument("-route", dest="route", type=str, default='net_jkt-new/jkt-new.rou.xml  ', help="Route definition xml file.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.0001, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.95, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.01, required=False, help="Epsilon.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=5, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=25, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=82800, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    args = prs.parse_args()

    out_csv = 'output/dqnbootsjkt_boots_expo_alfa_testing_final2_head2'
    cfg_file_='net_jkt-new/jkt-new.sumocfg' #nets2x2/2x2.sumocfg
    out_csv2 = 'output/simulationvalue2.csv'
    runs = 1
    batch_size = 256
    num_heads = 3

    env = SumoEnvironmentPZ(net_file='net_jkt-new/jkt-new.net.xml', #nets2x2/2x2.net.xml
                          #single_agent=True,
                          cfg_file=cfg_file_,
                          route_file=args.route,
                          out_csv_name=out_csv,
                          out_csv_name2 = out_csv2,
                          use_gui=False,
                          num_seconds=args.seconds,
                          min_green=args.min_green,
                          max_green=args.max_green)

    dqn_agent = {ts: DQNAgentBoostrapped(state_size=env.observation_space(ts).shape[0],action_size=env.action_space(ts).n) for ts in env.agents}
    '''
    for agent in env.agents:
        dqn_agent[agent].load('output-agent-boots-2/28800-324000-head1/'+str(agent)+'__'+str(25)+'.h5',
                                    'output-agent-boots-2/28800-324000-head2/'+str(agent)+'__'+str(25)+'.h5',
                                    'output-agent-boots-2/28800-324000-head3/'+str(agent)+'__'+str(25)+'.h5')
    '''

    for run in range(1, runs+1):
        env.reset()
        state = {ts: env.observe(ts) for ts in env.agents}
        dqn_agent = {ts: DQNAgentBoostrapped(state_size=env.observation_space(ts).shape[0],action_size=env.action_space(ts).n) for ts in env.agents}
        memories = {ts: Memory(10000) for ts in env.agents}
        memories_head = {ts: MemoryHeads(10000) for ts in env.agents}
        done = False
        reward_all = 0
        mask = deque(maxlen=1)
        s = []
        b = []
        size = {ts: s for ts in env.agents}
        batch = {ts: b for ts in env.agents}
        #print(size)
        random_state = np.random.RandomState(3)

        for ts in env.agents:
            dqn_agent[ts].choose_model()

        for agent in env.agent_iter():

            states_next, reward, done, info = env.last()

            #print(agent, states_next)

            exp_mask = random_state.binomial(1, 0.5, 3)

            mask.append(exp_mask)

            action = dqn_agent[agent].act(state[agent]) if not done else None

            env.step(action)

            memories[agent].remember(state=state[agent], action=action, reward=reward, state_next=states_next, done=done, mask=mask, num_agent=agent)

            if memories[agent].data[0][5][0][0] == 1:
                f = deepcopy(memories[agent].data)
                memories_head[agent].datahead1 = f

            if memories[agent].data[0][5][0][1] == 1:
                h = deepcopy(memories[agent].data)
                memories_head[agent].datahead2 = h

            if memories[agent].data[0][5][0][2] == 1:
                m = deepcopy(memories[agent].data)
                memories_head[agent].datahead3 = m

            if memories_head[agent].pointer1 < len(memories_head[agent].datahead1):
                memories_head[agent].pointer1 += 1

            if memories_head[agent].pointer2 < len(memories_head[agent].datahead2):
                memories_head[agent].pointer2 += 1

            if memories_head[agent].pointer3 < len(memories_head[agent].datahead3):
                memories_head[agent].pointer3 += 1
            #print(memories[agent].data[0]) #pertama batch, kedua state/action dll., ketiga buat manggil didalem state/action dll.

            size[agent] = [memories_head[agent].pointer1, memories_head[agent].pointer2, memories_head[agent].pointer3]

            #print(size[agent])

            #print(size['1'], memories_head['1'].pointer1, memories_head['1'].pointer2, memories_head['1'].pointer3)

            #print(memories_head['1'].pointer1, len(memories_head['1'].datahead1), memories_head['1'].pointer2, len(memories_head['1'].datahead2), memories_head['1'].pointer3, len(memories_head['1'].datahead3))

            batch[agent] = [random.sample(range(size[agent][0]), size[agent][0]) if size[agent][0] < batch_size else random.sample(range(size[agent][0]), batch_size),
                            random.sample(range(size[agent][1]), size[agent][1]) if size[agent][1] < batch_size else random.sample(range(size[agent][1]), batch_size),
                            random.sample(range(size[agent][2]), size[agent][2]) if size[agent][2] < batch_size else random.sample(range(size[agent][2]), batch_size)]


            #batch1 = random.sample(range(size[agent][0]), size[agent][0]) if size[agent][0] < batch_size else random.sample(range(size[agent][0]), batch_size)
            #batch2 = random.sample(range(size[agent][1]), size[agent][1]) if size[agent][1] < batch_size else random.sample(range(size[agent][1]), batch_size)
            #batch3 = random.sample(range(size[agent][2]), size[agent][2]) if size[agent][2] < batch_size else random.sample(range(size[agent][2]), batch_size)
            #print(len(batch['1'][0]), size['1']))

            #size3 = memories[agent].pointer
            #batch3 = random.sample(range(size3), size3) if size3 < batch_size else random.sample(range(size3), batch_size)

            #print(batch3, batch['1'])
            #print(memories_head['5'].datahead1, memories_head['5'].datahead2, memories_head['5'].datahead3)
            #if not batch[agent][0]:
            #    pass
            #else:
            #    print(batch[agent][0], batch3)
            #print(size[agent], size3)
            #print(memories_head[agent].datahead1)
            #print(size3, batch3)
            #print(memories_head['1'].datahead1)
            #print(memories['1'].data)
            #print(batch['1'])
            #print(*memories[agent].sample(batch3))
            #if len(memories_head[agent].datahead1) > batch_size:
            #    dqn_agent[agent].replay1(*memories_head[agent].sample1(batch3))
            #if len(memories_head[agent].datahead2) > batch_size:
            #    dqn_agent[agent].replay2(*memories_head[agent].sample2(batch3))
            #if len(memories_head[agent].datahead3) > batch_size:
            #    dqn_agent[agent].replay3(*memories_head[agent].sample3(batch3))

            if len(memories_head[agent].datahead1) > batch_size:
                if not batch[agent][0]:
                    pass
                else:
                    dqn_agent[agent].replay1(*memories_head[agent].sample1(batch[agent][0]))
                    #print(*memories_head[agent].sample1(batch[agent][0]))

            if len(memories_head[agent].datahead2) > batch_size:
                if not batch[agent][1]:
                    pass
                else:
                    dqn_agent[agent].replay2(*memories_head[agent].sample2(batch[agent][1]))
            if len(memories_head[agent].datahead3) > batch_size:
                if not batch[agent][2]:
                    pass
                else:
                    dqn_agent[agent].replay3(*memories_head[agent].sample3(batch[agent][2]))


            #print(size, len(memories_head[agent].datahead1), len(memories[agent].data))
            #print(len(memories_head[agent].datahead1), len(memories_head['1'].datahead1), len(memories_head['2'].datahead1), len(memories_head['5'].datahead1), len(memories_head['6'].datahead1))
            #print(memories_head[agent].datahead1)
            #print(*memories_head[agent].sample1(batch))
            #print(*memories[agent].data)
            #print(*memories_head[agent].datahead1)
            #print(memories_head[agent].datahead3)


            #print(*memories_head['2'].datahead1)

            #print(*memories['1'].data)
            #print(*memories_head.datahead1)

            #print(*memories[agent].sample(batch))

            state[agent] = states_next

            reward_all += reward
            '''
            if done : #save nn weight if training done
                dqn_agent[agent].save('output-agent/output-agent-boots-2/28800-324000-head1/'+str(agent)+'__'+str(run)+'.h5',
                                    'output-agent/output-agent-boots-2/28800-324000-head2/'+str(agent)+'__'+str(run)+'.h5',
                                    'output-agent/output-agent-boots-2/28800-324000-head3/'+str(agent)+'__'+str(run)+'.h5')
            '''


        print("Score: {s}, Goal: {g}".format(s=reward_all, g=done))

        env.unwrapped.env.save_csv(out_csv, run)
        env.close()
