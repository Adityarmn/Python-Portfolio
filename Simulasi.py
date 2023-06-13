import argparse
import os
import sys
from environment.sumo_petting_zoo import SumoPettingZoo
from dqn_marl import DQNAgent, Memory
import random



def simulasi(alpha, density):

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
    prs.add_argument("-start", dest="start", type=int, default=8*3600, required=False, help="Start Simulation.\n")
    prs.add_argument("-finish", dest="finish", type=int, default=9*3600, required=False, help="Finish Simulation.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    prs.add_argument("-alpha", dest="alpha", type=float, default=alpha, help="alpha\n")
    prs.add_argument("-density", dest="density", type=float, default=density, help="density\n")

    args = prs.parse_args()

    out_csv = 'output/dqn_jkt_test_versi2_final_2'
    cfg_file_='net_jkt-new/jkt-new.sumocfg'
    runs = 1
    batch_size = 256
    
    env = SumoPettingZoo(net_file='net_jkt-new/jkt-new.net.xml',
                          #single_agent=True,
                          route_file=args.route,
                          cfg_file = cfg_file_,
                          out_csv_name=out_csv,
                          use_gui=False,
                          begin_time=args.start,
                          num_seconds=args.finish,
                          min_green=args.min_green,
                          max_green=args.max_green)


    #dqn_agent = {ts: DQNAgent(state_size=env.observation_space(ts).shape[0],action_size=env.action_space(ts).n) for ts in env.agents}

    #for ts in env.agents:
     #   dqn_agent[ts].load('output-training/57600-61200-dqn/'+str(agent)+'__'+str(run)+'.h5')
      
    for run in range(1, runs+1): #diganti-ganti mau sekali running udah atau langsung banyak
        env.reset()
        state = {ts: env.observe(ts) for ts in env.agents}
        dqn_agent = {ts: DQNAgent(state_size=env.observation_space(ts).shape[0],action_size=env.action_space(ts).n) for ts in env.agents}
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

        env.unwrapped.env.save_csv(out_csv, run)
        env.close()
        
        if done : #save nn weight if training done
            dqn_agent[agent].save('output-training/57600-61200-dqn/'+str(agent)+'__'+str(run)+'.h5')
        

    # Buat Objective Function, datanya diambil dari excel hasil test
        L1_min = 10 
        L2_max = 13
        L4_max =14

        J = L2_max + L4_max - L1_min
    return J