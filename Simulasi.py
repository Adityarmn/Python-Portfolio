import argparse
import os
import sys
import pandas as pd
from sumo_petting_zoo import SumoPettingZoo
from dqn_marl import DQNAgent, Memory
import random

def validasi(beta, density, time_start, time_end, runs = 1):

    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("Please declare the environment variable 'SUMO_HOME'")
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    prs.add_argument("-route", dest="route", type=str, default='net_jkt-new/jkt-new.rou.xml', help="Route definition xml file.\n") #netsingle/single.rou.xmlnewnetwork/jkt-new.rou.xml
    prs.add_argument("-a", dest="alpha", type=float, default=0.0001, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.95, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.01, required=False, help="Epsilon.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=5, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=25, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-start", dest="start", type=int, default=time_start*3600, required=False, help="Start Simulation.\n")
    prs.add_argument("-finish", dest="finish", type=int, default=time_end*3600, required=False, help="Finish Simulation.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=runs, help="Number of runs.\n")
    prs.add_argument("-beta", dest="beta", type=float, default=beta, help="beta\n")
    prs.add_argument("-density", dest="density", type=float, default=density, help="density\n")

    args = prs.parse_args()

    out_xml = 'Iterasi'
    cfg_file_='net_jkt-new/jkt-new.sumocfg'
    # runs = 1
    batch_size = 256
    
    env = SumoPettingZoo(net_file='net_jkt-new/jkt-new.net.xml',
                          #single_agent=True,
                          route_file=args.route,
                          cfg_file = cfg_file_,
                          out_xml_name=out_xml,
                          use_gui=False,
                          begin_time=args.start,
                          num_seconds=args.finish,
                          min_green=args.min_green,
                          max_green=args.max_green)


    dqn_agent = {ts: DQNAgent(state_size=env.observation_space(ts).shape[0],action_size=env.action_space(ts).n) for ts in env.agents}

    # for ts in env.agents:
    #  dqn_agent[ts].load('output-training\tesaja'+str(agent)+'__'+str(run+33)+'.h5')
      
    for run in range(1, runs+1): #diganti-ganti mau sekali running udah atau langsung banyak
        env.reset()
        state = {ts: env.observe(ts) for ts in env.agents}
        # dqn_agent = {ts: DQNAgent(state_size=env.observation_space(ts).shape[0],action_size=env.action_space(ts).n) for ts in env.agents}
        memories = {ts: Memory(10000) for ts in env.agents}
        done = False
        reward_all = 0

        for agent in env.agent_iter():

            observation, reward, termination, truncation, info = env.last()

            action = None if termination or truncation else dqn_agent[agent].act(state[agent])  # this is where you would insert your policy

            env.step(action)

            action_size = dqn_agent[agent].action_size
 
            memories[agent].remember(state=state[agent], action=action, reward=reward, state_next=observation, done=done)

            size = memories[agent].pointer
            batch = random.sample(range(size), size) if size < batch_size else random.sample(range(size), batch_size)

            if len(memories[agent].data) > batch_size:
                dqn_agent[agent].replay(*memories[agent].sample(batch))

            state[agent] = observation

            reward_all += reward

        print("Score: {s}, Goal: {g}".format(s=reward_all, g=done))

        
        # data = env.unwrapped.env.save_csv(out_csv, run)
        # data = pd.read_xml("net_jkt-new/output/lane-data.xml")
        print("--------------")
        print(run)
        env.close()
        
        if done : #save nn weight if training done
                dqn_agent[agent].save('output-running'+str(agent)+'__'+str(run)+'.h5')


        # Operasi MFD
        # Data dari MFD   
        # 1. Load data csv (Hasil dari testing)
        # 2. Hitung MFD per episode
        # 3. Tentuin area L1 L2 L3 L4 
        # 4. Tentuin nilai Min/Max sesuai kebutuhan dari masing-masing area

        # Buat Objective Function, datanya diambil dari excel hasil test
        L1_min = 10
        L2_max = 13
        L3_min = 12
        L4_max = 14

            # Buat rumus bisa dibikin kuadratik, trus tiap variabel ditambahin beban random P,Q,R yang rangenya 0 sampai 1
            # buat beban jadiin 1 dulu biar gampang
        P = 1 
        Q = 1 
        R = 1
        S = 1
            
        J = (P * (L2_max**2)) + (Q * L4_max**2) - (R * L1_min) - (S * L3_min) #tambahin L3 dan rumus objective functionnya masi bisa berubah-ubah
