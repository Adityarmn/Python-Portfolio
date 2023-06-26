

import matplotlib.pyplot as plt
import pandas as pd
import datetime

csv_lanelength = 'dwdwd'
csv_list = ''

def get_lane_length(csv_lanelength):
    lanelength_data = pd.read_csv(csv_lanelength, skiprows=1, header=None)
    lane_list = lanelength_data[0]
    length_list = lanelength_data[1]/1000
    #Simpan di dictionary:
    lane_length = dict(zip(lane_list,length_list))
    return lane_length

def get_data_group(csv_flow_and_density): #grouping data ke pandas object
    #get flow(veh/hr) and density(veh/km) di setiap lane untuk interval tertentu
    flowdensity = pd.read_csv(csv_flow_and_density, header = 0)
    
    #Mengambil list waktu('time') di mfd-data.csv
    flowdensity_lane = flowdensity.groupby('lane_ids')
    dummy_group = flowdensity_lane.get_group('1_0')
    time_list = dummy_group['time']

    #grouping menggunakan waktu real
    flowdensity_time = flowdensity.groupby('time')
    return time_list, flowdensity_time

def get_MFD_property(time_list, flowdensity_time, total_lane_length, lane_length_dict ): #menghitung Qt dan Kt
    Q = [] #List network flow
    K = [] #List network density

    #Menghitung network flow (Qt) dan network density (Kt)
    for time in time_list[::]:
        the_group = flowdensity_time.get_group(time)
        lane_ids = the_group['lane_ids']
        lane_flows = the_group['outflow']
        lane_densities = the_group['laneDensity']
        flowx = []
        densityx = []
        
        # Ga harus pakai enumerate untuk benerin datanya
        for i,lane in enumerate(lane_ids):
            lane_id = lane_ids.iloc[i]
            lane_flow = lane_flows.iloc[i]
            lane_density = lane_densities.iloc[i]
            lane_length = lane_length_dict[str(lane_id)] #get lane length from earlier dictionary in lane-length.csv
        
            flowxlength = lane_flow*lane_length
            flowx.append(flowxlength) #list
            densityxlength = lane_density*lane_length
            densityx.append(densityxlength) #list
        Qt = sum(flowx)/total_lane_length
        Kt = sum(densityx)/total_lane_length
        Q.append(Qt)
        K.append(Kt)
        
