import argparse
import multiprocessing
import os
import sys
import time, pyautogui
import PySimpleGUI as sg
from datetime import datetime

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from env_baru import SumoEnvironmentPZ
from env_baru import SumoEnvironment
from dqn_marl import DQNAgent
from dqn_marl import Memory
import numpy as np
from pathlib import Path
import csv
import pandas as pd
import random
if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #prs.add_argument("-route", dest="route", type=str, default='nets/single-intersection.rou.xml', help="Route definition xml file.\n")
    prs.add_argument("-route", dest="route", type=str, default='net_jkt-new/jkt-new.rou.xml', help="Route definition xml file.\n") #netsingle/single.rou.xmlnewnetwork/jkt-new.rou.xml
    prs.add_argument("-a", dest="alpha", type=float, default=0.0001, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.95, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.01, required=False, help="Epsilon.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=5, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=25, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=41600, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    args = prs.parse_args()

    out_csv = 'output/dqn_jkt_test_0.1_0.9_max20_ujikedua_4'
    cfg_file_='net_jkt-new/jkt-new.sumocfg'#net-single/single.sumocfg
    out_csv2 = 'output/simulationvalue1.csv'
    runs = 1
    batch_size = 256

    env = SumoEnvironmentPZ(net_file='net_jkt-new/jkt-new.net.xml', #net-single/single.net.xml
                          #single_agent=True,
                          cfg_file=cfg_file_,
                          route_file=args.route,
                          out_csv_name=out_csv,
                          out_csv_name2 = out_csv2,
                          use_gui=False,
                          num_seconds=args.seconds,
                          min_green=args.min_green,
                          max_green=args.max_green)

    #env.reset()
    #state={ts: env.observe(ts) for ts in env.agents}
    #print(state)cd
    #for ts in env.agents:
        #print(ts)

#state = {ts: env.observe(ts) for ts in env.agents}
    '''
    dqn_agent = {ts: DQNAgent(state_size=env.observation_space(ts).shape[0],action_size=env.action_space(ts).n) for ts in env.agents}

    for ts in env.agents:
        dqn_agent[ts].load('output-agent/41400-45000-dqn-0.1-0.9_max20_2/'+str(ts)+'__'+str(33)+'.h5')
    '''
    for run in range(1, runs+1):
        env.reset()
        state = {ts: env.observe(ts) for ts in env.agents}
        dqn_agent = {ts: DQNAgent(state_size=env.observation_space(ts).shape[0],action_size=env.action_space(ts).n) for ts in env.agents}
        memories = {ts: Memory(10000) for ts in env.agents}
        done = False
        reward_all = 0
        #dfs = []
        #print(env.num_agents)
        #b = [2,3]
        #c = memories['0'].data
        #c.append(b)
        #print(c)
        #actions = []
        for agent in env.agent_iter():

            states_next, reward, done, info = env.last()

            print(reward)

            action = dqn_agent[agent].act(state[agent]) if not done else None

            env.step(action)

            action_size = dqn_agent[agent].action_size
            #print(state[agent])
            '''
            density = state[agent][action_size+1:action_size*2+1]
            queue = state[agent][action_size*2+1:]
            write_list1 = [str(agent), density]
            write_list2 = [str(agent), queue]
            with open('test_density.csv', 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f,lineterminator = '\n')
                writer.writerow(write_list1)
            with open('test_queue.csv', 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f,lineterminator = '\n')
                writer.writerow(write_list2)
            '''

            #print(density)
            #print(queue)
            #actions.append(action)
            memories[agent].remember(state=state[agent], action=action, reward=reward, state_next=states_next, done=done)

            #print(memories[agent].data[0]) #pertama batch, kedua state/action dll., ketiga buat manggil didalem state/action dll.
            #print(len(memories[agent].data))

            size = memories[agent].pointer
            batch = random.sample(range(size), size) if size < batch_size else random.sample(range(size), batch_size)

            if len(memories[agent].data) > batch_size:
                dqn_agent[agent].replay(*memories[agent].sample(batch))


            #filepath = Path('net-single/single_out.csv')
            #filepath.parent.mkdir(parents=True, exist_ok=True)

            #df = dqn_agent[agent].action_values
            #df1 = pd.DataFrame(df.reshape(-1,len(df)), columns = ['action0','action1','action2','action3'])
            #dfs.append(df1)
            #with open('net-single/single-net.csv', 'w', encoding='UTF8', newline='') as f:
            #print(dqn_agent[agent].action_values)
                #writer = csv.writer(f)
                #writer.writerows(dqn_agent[agent].action_values)

            state[agent] = states_next

            reward_all += reward

        #    if done : #save nn weight if training done
        #        dqn_agent[agent].save('output-agent/41400-45000-dqn-0.1-0.9_max20_2/'+str(agent)+'__'+str(run+33)+'.h5')

        print("Score: {s}, Goal: {g}".format(s=reward_all, g=done))

        env.unwrapped.env.save_csv(out_csv, run)
        env.close()

        #    print(memories)
        #    if not args.fixed:
        #        size = memories[0].pointer
        #        batch = random.sample(range(size), size) if size < batch_size else random.sample(range(size), batch_size)

        #        for agent in env.agent_iter():
        #            memories[agent].remember(states[agent], actions[agent], rewards[agent], states_next[agent], done[agent])

        #            if memories[i].pointer > batch_size * 10:
        #                dqn_agent[i].replay(*memories[i].sample(batch))

        #    states = states_next
        #    reward_all += reward

    #    reward_list.append(reward_all)

    #    print("Score: {s}, Goal: {g}".format(s=reward_all, g=done))

    #    env.unwrapped.env.save_csv(out_csv, run)
    #    env.close()


# Dibawah ini kode buat single agent

    #for run in range(1, args.runs+1):
    #    state = env.reset()
    #    agent = DQNAgent(state_size, action_size)

    #    done = False
    #    if args.fixed:
    #        while not done:
    #            _, _, done, _ = env.step({})

    #    else:
    #        while not done:
    #            action = agent.act(state)

    #            next_state, r, done, _ = env.step(action=action)

    #            agent.memorize(state=state, action=action, reward=r, next_state=next_state, done=done)

    #            if done:
    #                print("episode: {}/{}, score: {}, e: {:.2}"
    #                    .format(e, EPISODES, time, agent.epsilon))
    #                break
    #            if len(agent.memory) > batch_size:
    #                agent.replay(batch_size)

    #            state = next_state
    #            env.write_csv()

    #    env.save_csv(out_csv, run)
def KeepUI():
    
    sg.theme('Dark')
    layout = [
        [sg.Text('Keep-Me-Up is now running.\nYou can keep it minised, and it will continue running.\nClose it to disable it.')]
    ]
    window = sg.Window('Keep-Me-Up', layout)
    
    p2 = multiprocessing.Process(target = dontsleep)
    p2.start()
    
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED: # if user closes window or clicks cancel
            if p2.is_alive(): 
                p2.terminate()
            break

def dontsleep():
    while True:
        pyautogui.press('volumedown')
        time.sleep(1)
        pyautogui.press('volumeup')
        time.sleep(300)



if __name__ == '__main__':
    p1 = multiprocessing.Process(target = KeepUI)
    p1.start()