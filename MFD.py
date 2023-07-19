import pandas as pd
import xml.etree.ElementTree as ET

def MFD():
    #Bikin program buat ngitung Ppeak Xpeak Pgridlock Xgridlock
    # 1. Load data XML pakai pandas read_xml dari folder net_jkt-new/output/lane-data
    # 2. Bikin MFD jadi data Kn Qn (flow banding density)
    # 3. Hitung Qpeak dll
    
    # Define a list to store the data
    data = []
    tree = ET.parse('net_jkt-new/output/lane-data.xml')
    root = tree.getroot()

    # Iterate over all "interval" elements in the root
    for interval in root:
        # Get the attributes of the interval
        interval_attribs = interval.attrib
        
        # Iterate over all "edge" elements in the interval
        for edge in interval:
            # Get the attributes of the edge
            edge_attribs = edge.attrib
            
            # Iterate over all "lane" elements in the edge
            for lane in edge:
                # Get the attributes of the lane
                lane_attribs = lane.attrib
                
                # Combine the attributes of the interval, edge, and lane
                combined_attribs = {**interval_attribs, **edge_attribs, **lane_attribs}
                
                # Append the combined attributes to the data list
                data.append(combined_attribs)

    # Convert the data list to a DataFrame
    df = pd.DataFrame(data)

    df['departed'] = df['departed'].astype(int)
    df['entered'] = df['entered'].astype(int)
    df['arrived'] = df['arrived'].astype(int)
    df['left'] = df['left'].astype(int)

    df['inflow'] = (df['departed'] + df['entered']) * 12
    df['outflow'] = (df['arrived'] + df['left']) * 12

    df = df[['begin', 'id', 'laneDensity', 'inflow', 'outflow']]

def get_MFD_property(time_list, flowdensity_time,total_lane_length,lane_length_dict): #calculate Qt and Kt
    Q = [] #list of network flow
    K = [] #list of network density
    #calculate network flow (Qt) and network density(Kt)
    for time in time_list[::]:
        the_group = flowdensity_time.get_group(time)
        lane_ids = the_group['lane_ids']
        lane_flows = the_group['outflow']
        lane_densities = the_group['laneDensity']
        flowx = []
        densityx= []
        #bisa dibenerin tanpa enumerate 
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
    return Q, K


    Qpeak = 20
    Kpeak = 10
    Qgridlock = 5
    Kgridlock = 25
    return Qpeak, Kpeak, Qgridlock, Kgridlock