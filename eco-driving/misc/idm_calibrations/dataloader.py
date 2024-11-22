import pandas as pd

FREQ = 30

lane_type = {(5, 6, 7, 17, 18, 19, 3, 4, 8): 'right',
             (12, 13, 14, 15, 16, 26, 27, 28): 'up',
             (0, 1, 2, 22, 23, 24, 20, 21, 25): 'left',
             (29, 30, 31, 32, 33, 34, 9, 10, 11): 'down'}

lane_dict = {lane: t for lanes, t in lane_type.items() for lane in lanes}

considered_lanes = [3]

def extract_info(data: pd.DataFrame, v_type: str):
    """
    Extracts the information from the csv file
    """
    mph_ms_convert_ratio = 0.44704 # convert from mph to m/s
    ft_m_convert_ratio = 0.3048 # convert from feet to m

    data_count = 0

    cars = 0
    trucks = 0
    v_ids = []
    for cnt, (index, group) in enumerate(data.groupby(['frameNum', 'laneId'])):
        _, lane_id = index
        # there are only 34 lanes in total at Alafaya intersection
        if lane_id > 34:
            continue
        # direction: left, right, up and down
        direction = lane_dict[lane_id]
        # only consider vehicles traveling in a certain direction on selected set of lanes
        if direction != "right" and lane_id not in considered_lanes:
            continue
        else:
            # sort vehicles based on their head X location
            sorted_vehicles = group.sort_values(by=['headXft'])
            # remove duplicate entries
            sorted_vehicles = sorted_vehicles.drop_duplicates(subset=['carId'])

            for i in range(len(sorted_vehicles) - 1): # ignore the last vehicle as it has no leader
                vehicle_length = sorted_vehicles.iloc[i]['headXft'] - sorted_vehicles.iloc[i]['tailXft']
                # assumptions: Cars are 5m long at the most (reference: Treiber's book)
                if vehicle_length*ft_m_convert_ratio < 5.5:
                    cars += 1 
                    if v_type == "TRUCKS":
                        continue
                    v_ids.append(sorted_vehicles.iloc[i]['carId'])
                else:
                    trucks += 1
                    if v_type == "CARS":
                        continue
                    v_ids.append(sorted_vehicles.iloc[i]['carId'])
                    
                idx_1 = sorted_vehicles.iloc[i]['index']
                # relative speed = ego vehicle speed - leading vehicle speed (note: the order matters)
                relative_speed = sorted_vehicles.iloc[i]['speed'] - sorted_vehicles.iloc[i+1]['speed']
                speed = sorted_vehicles.iloc[i]['speed']
                # headway = leading vehicle tail X location - ego vehicle head X location
                headway = sorted_vehicles.iloc[i+1]['tailXft'] - sorted_vehicles.iloc[i]['headXft']
                # modify the data data frame with additional features
                data.loc[idx_1, 'relative_speed'] = relative_speed * mph_ms_convert_ratio
                data.loc[idx_1, 'speed'] = speed * mph_ms_convert_ratio
                data.loc[idx_1, 'headway'] = headway * ft_m_convert_ratio
                data_count += 1
                
        print(cnt, end='\r')

    print(f"Cars: {cars} Trucks: {trucks}")
    # drop rows if there are any rows having NaNs for required features (non-relevant data)
    data.dropna(inplace=True, subset=["speed", "headway", "relative_speed"])
    print(f"Data count: {data_count}")
    return data, v_ids

def extract_car_info(data: pd.DataFrame, car_ids):
    """
    Extracts data about a given car from the csv file
    """
    ego_velocities = {}
    ego_heads = {}
    leader_velocities = {}
    leader_tails = {}
    cars = []

    mph_ms_convert_ratio = 0.44704 # convert from mph to m/s
    ft_m_convert_ratio = 0.3048 # convert from feet to m

    for cnt, (index, group) in enumerate(data.groupby(['frameNum', 'laneId'])):
        _, lane_id = index
        # there are only 34 lanes in total at Alafaya intersection
        if lane_id > 34:
            continue
        # direction: left, right, up and down
        direction = lane_dict[lane_id]
        # only consider vehicles traveling in a certain direction on selected set of lanes
        if direction != "right" and lane_id not in considered_lanes:
            continue
        else:
            # sort vehicles based on their head X location
            sorted_vehicles = group.sort_values(by=['headXft'])
            # remove duplicate entries
            sorted_vehicles = sorted_vehicles.drop_duplicates(subset=['carId'])
            
            for i in range(len(sorted_vehicles)):
                if sorted_vehicles.iloc[i]['carId'] not in cars:
                    cars.append(sorted_vehicles.iloc[i]['carId'])
                    continue
                else:
                    e_speed = sorted_vehicles.iloc[i]['speed'] * mph_ms_convert_ratio
                    e_head = sorted_vehicles.iloc[i]['headXft'] * ft_m_convert_ratio
                    # what if there is no leader?
                    if i == len(sorted_vehicles) - 1:
                        l_speed = 20 # speed limit of the road
                        l_tail = e_head + 100 # 100 from the ego-vehicle
                    else:
                        l_speed = sorted_vehicles.iloc[i+1]['speed'] * mph_ms_convert_ratio
                        l_tail = sorted_vehicles.iloc[i+1]['tailXft'] * ft_m_convert_ratio

                    car_id = sorted_vehicles.iloc[i]['carId']
                    if car_id not in ego_velocities.keys():
                        ego_velocities[car_id] = [e_speed]
                        ego_heads[car_id] = [e_head]
                        leader_velocities[car_id] = [l_speed]
                        leader_tails[car_id] = [l_tail]
                    else:
                        ego_velocities[car_id].append(e_speed)
                        ego_heads[car_id].append(e_head)
                        leader_velocities[car_id].append(l_speed)
                        leader_tails[car_id].append(l_tail)
            
        print(cnt, end='\r')
    return ego_velocities, ego_heads, leader_velocities, leader_tails