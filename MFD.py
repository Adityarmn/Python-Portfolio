import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET


def get_data_group(df_flow_and_density):  # grouping data to pandas object
    # get flow(veh/hr) and density(veh/km) in each lane for certain interval
    flowdensity = df_flow_and_density

    # only for get a list of 'time' in mfd-data.csv
    flowdensity_lane = flowdensity.groupby("lane_ids")
    dummy_group = flowdensity_lane.get_group("1_1")
    time_list = dummy_group["time"]

    # real grouping (group by time)
    flowdensity_time = flowdensity.groupby("time")
    return time_list, flowdensity_time


def get_MFD_property(
    time_list, flowdensity_time, total_lane_length, lane_length_dict
):  # calculate Qt and Kt
    Q = []  # list of network flow
    K = []  # list of network density
    # calculate network flow (Qt) and network density(Kt)
    for time in time_list:
        the_group = flowdensity_time.get_group(time)
        lane_ids = the_group["lane_ids"]
        lane_flows = the_group["outflow"]
        lane_densities = the_group["laneDensity"]
        flowx = []
        densityx = []
        # bisa dibenerin tanpa enumerate
        for i, lane in enumerate(lane_ids):
            lane_id = lane_ids.iloc[i]
            lane_flow = lane_flows.iloc[i]
            lane_densities = lane_densities.astype(float)
            lane_density = lane_densities.iloc[i]
            lane_length = lane_length_dict[
                str(lane_id)
            ]  # get lane length from earlier dictionary in lane-length.csv

            flowxlength = lane_flow * lane_length
            flowx.append(flowxlength)  # list
            densityxlength = lane_density * lane_length
            densityx.append(densityxlength)  # list
        Qt = sum(flowx) / total_lane_length
        Kt = sum(densityx) / total_lane_length
        Q.append(Qt)
        K.append(Kt)
    return Q, K


def get_lane_length(df_lanelength):  # get lane length information
    lanelength_data = pd.read_csv(df_lanelength, skiprows=1, header=None)
    lane_list = lanelength_data[0]
    length_list = lanelength_data[1] / 1000
    # store in dictionary :
    lane_length = dict(zip(lane_list, length_list))
    return lane_length


def MFD():
    # Bikin program buat ngitung Ppeak Xpeak Pgridlock Xgridlock
    # 1. Load data XML pakai pandas read_xml dari folder net_jkt-new/output/lane-data
    # 2. Bikin MFD jadi data Kn Qn (flow banding density)
    # 3. Hitung Qpeak dll

    # Define a list to store the data
    time_list = {}
    flowdensity_time = {}
    Qn = {}
    Kn = {}

    lane_length_dict = get_lane_length("lane-length-new.csv")
    total_lane_length = sum(
        lane_length_dict.values()
    )  # total lane length in network (KM)

    data = []
    tree = ET.parse("net_jkt-new/output/lane-data.xml")
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

    df["departed"] = df["departed"].astype(int)
    df["entered"] = df["entered"].astype(int)
    df["arrived"] = df["arrived"].astype(int)
    df["left"] = df["left"].astype(int)

    df["inflow"] = (df["departed"] + df["entered"]) * 12
    df["outflow"] = (df["arrived"] + df["left"]) * 12

    df = df[["begin", "id", "laneDensity", "inflow", "outflow"]]

    # Rename "begin" column to "time" and id column to "lane_ids"
    df = df.rename(columns={"begin": "time", "id": "lane_ids"})
    df = df.dropna()

    time_list, flowdensity_time = get_data_group(df)
    Qn, Kn = get_MFD_property(
        time_list, flowdensity_time, total_lane_length, lane_length_dict
    )
    
    # Qpeak itu nilai maks sumbu y dari MFD tapi dari sumbu y itu yang dipake sumbu x nya, Q gridlock itu nilai maks sumbu x dari MFD,
    Qpeak = np.max(Qn)  # di dalem kurung itu nilai maksimum dari mfd
    Kpeak = Kn[np.argmax(Qn)]
    Qgridlock = np.max(Kn)  # Nilai ini diambil dari titik paling kanan di mfd
    Kgridlock = Qn[np.argmax(Kn)]

    return Qpeak, Kpeak, Qgridlock, Kgridlock
