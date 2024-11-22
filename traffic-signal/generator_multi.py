import numpy as np
import math
import shutil
from pathlib import Path
import os
import re
from xml.etree import ElementTree


class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated, num_lanes, lane_length, speed_limit, left_turn, alg, trial, dir_name):
        # how many cars per episode
        self._n_cars_generated = n_cars_generated
        self._max_steps = max_steps
        self._num_lanes = num_lanes
        self._lane_length = lane_length
        self._speed_limit = speed_limit
        self._left_turn = left_turn
        self._alg = alg
        self._trial = trial
        self._dir_name = dir_name
        # self._dir_name = f"intersection_flow{self._n_cars_generated}_lane{self._num_lanes}_length{self._lane_length}_speed{self._speed_limit}_left{self._left_turn}_alg{self._alg}_trial{self._trial}"

        
    def generate_netfile(self, seed):
        """
        Generation of the netfile for one episode
        """
        np.random.seed(seed)
        
        # define the network geometry
        
        # base directory
        base_dir = "./network/base_intersection/"
        
        # copy whole base directory to the current directory including the subdirectories and files
        dir_name = self._dir_name
        if os.path.exists(f"./results/{dir_name}/network"):
            shutil.rmtree(f"./results/{dir_name}/network", ignore_errors=True)
        shutil.copytree(base_dir, f"./results/{dir_name}/network")
        # if not os.path.exists(f"./results/{dir_name}/network"):
        #     shutil.copytree(base_dir, f"./results/{dir_name}/network")
        # print(f"Copied base directory to ./results/{dir_name}")

        
        # get input about lane length, lane numbers, and traffic light phases
        net_raw = (open(str(Path(f"./results/{dir_name}/network/environment.net.xml")), 'r')
            .read()
            .replace('13.89', str(self._speed_limit)))

        text = re.split('\[.{6,7}\]', net_raw)
        numbers = [str(float(token[1:-1]) + (1 if float(token[1:-1]) > 0 else -1) * (self._lane_length - 750))
                   for token in re.findall('\[.{6,7}\]', net_raw)]

        net_raw = ''.join(token
                          for tokens in zip(text, numbers)
                          for token in tokens) + text[-1]

        open(str(Path(f"./results/{dir_name}/network/environment.net.xml")), 'w').write(net_raw)


    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible
        dir_name = self._dir_name
        left_turn = self._left_turn
        speed_limit = self._speed_limit

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps,
                                      ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        # round every value to int -> effective steps when a car will be generated
        car_gen_steps = np.rint(car_gen_steps)

        # produce the file for cars generation, one car per line
        with open(f"./results/{dir_name}/network/episode_routes.rou.xml", "w") as routes:
            print("""<routes>
    <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" speedFactor="norm(1,0)"/>

    <route id="W_N" edges="W2TL TL2N"/>
    <route id="W_E" edges="W2TL TL2E"/>
    <route id="W_S" edges="W2TL TL2S"/>
    <route id="N_W" edges="N2TL TL2W"/>
    <route id="N_E" edges="N2TL TL2E"/>
    <route id="N_S" edges="N2TL TL2S"/>
    <route id="E_W" edges="E2TL TL2W"/>
    <route id="E_N" edges="E2TL TL2N"/>
    <route id="E_S" edges="E2TL TL2S"/>
    <route id="S_W" edges="S2TL TL2W"/>
    <route id="S_N" edges="S2TL TL2N"/>
    <route id="S_E" edges="S2TL TL2E"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                # choose direction: straight or turn - 1-self._left_turn of times the car goes straight
                departspeed = speed_limit if speed_limit < 10 else 10
                if straight_or_turn < 1-left_turn:
                    # choose a random source & destination
                    route_straight = np.random.randint(1, 5)
                    
                    if route_straight == 1:
                        print('    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)
                    elif route_straight == 2:
                        print('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)
                    elif route_straight == 3:
                        print('    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)
                    else:
                        print('    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)
                else:  # self._left_turn of the time, the car turns
                    route_turn = np.random.randint(1, 9)  # choose random source source & destination
                    if route_turn == 1:
                        print('    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)
                    elif route_turn == 2:
                        print('    <vehicle id="W_S_%i" type="standard_car" route="W_S" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)
                    elif route_turn == 3:
                        print('    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)
                    elif route_turn == 4:
                        print('    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)
                    elif route_turn == 5:
                        print('    <vehicle id="E_N_%i" type="standard_car" route="E_N" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)
                    elif route_turn == 6:
                        print('    <vehicle id="E_S_%i" type="standard_car" route="E_S" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)
                    elif route_turn == 7:
                        print('    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)
                    elif route_turn == 8:
                        print('    <vehicle id="S_E_%i" type="standard_car" route="S_E" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)

            print("</routes>", file=routes)

class TrafficGeneratorTransfer(TrafficGenerator):
    def __init__(self, source_path_name, max_steps, n_cars_generated, num_lanes, lane_length, speed_limit, left_turn, alg, trial):
        self._source_path_name = source_path_name
        self._n_cars_generated = n_cars_generated
        self._max_steps = max_steps
        self._num_lanes = num_lanes
        self._lane_length = lane_length
        self._speed_limit = speed_limit
        self._left_turn = left_turn
        self._alg = alg
        self._trial = trial
        self._dir_name = f"intersection_flow{self._n_cars_generated}_lane{self._num_lanes}_length{self._lane_length}_speed{self._speed_limit}_left{self._left_turn}_alg{self._alg}_trial{self._trial}"

        
    def generate_netfile(self, seed):
        """
        Generation of the netfile for one episode
        """
        np.random.seed(seed)
        
        # base directory
        base_dir = "./network/base_intersection/"
        
        # copy whole base directory to the current directory including the subdirectories and files
        dir_name = self._dir_name
        source_path_name = self._source_path_name
        # shutil.copytree(base_dir, f"./results/{dir_name}/network")
        if not os.path.exists(f"./results/{source_path_name}/transfer/{dir_name}/network"):
            shutil.copytree(base_dir, f"./results/{source_path_name}/transfer/{dir_name}/network")
        # print(f"Copied base directory to ./results/{dir_name}")

        
        # get input about lane length, lane numbers, and traffic light phases
        net_raw = (open(str(Path(f"./results/{source_path_name}/transfer/{dir_name}/network/environment.net.xml")), 'r')
            .read()
            .replace('13.89', str(self._speed_limit)))

        text = re.split('\[.{6,7}\]', net_raw)
        numbers = [str(float(token[1:-1]) + (1 if float(token[1:-1]) > 0 else -1) * (self._lane_length - 750))
                   for token in re.findall('\[.{6,7}\]', net_raw)]

        net_raw = ''.join(token
                          for tokens in zip(text, numbers)
                          for token in tokens) + text[-1]

        open(str(Path(f"./results/{source_path_name}/transfer/{dir_name}/network/environment.net.xml")), 'w').write(net_raw)


    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible
        dir_name = self._dir_name
        source_path_name = self._source_path_name
        left_turn = self._left_turn
        speed_limit = self._speed_limit

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps,
                                      ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        # round every value to int -> effective steps when a car will be generated
        car_gen_steps = np.rint(car_gen_steps)

        # produce the file for cars generation, one car per line
        with open(f"./results/{source_path_name}/transfer/{dir_name}/network/episode_routes.rou.xml", "w") as routes:
            print("""<routes>
    <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" speedFactor="norm(1,0)"/>

    <route id="W_N" edges="W2TL TL2N"/>
    <route id="W_E" edges="W2TL TL2E"/>
    <route id="W_S" edges="W2TL TL2S"/>
    <route id="N_W" edges="N2TL TL2W"/>
    <route id="N_E" edges="N2TL TL2E"/>
    <route id="N_S" edges="N2TL TL2S"/>
    <route id="E_W" edges="E2TL TL2W"/>
    <route id="E_N" edges="E2TL TL2N"/>
    <route id="E_S" edges="E2TL TL2S"/>
    <route id="S_W" edges="S2TL TL2W"/>
    <route id="S_N" edges="S2TL TL2N"/>
    <route id="S_E" edges="S2TL TL2E"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                # choose direction: straight or turn - 1-self._left_turn of times the car goes straight
                departspeed = speed_limit if speed_limit < 10 else 10
                if straight_or_turn < 1-left_turn:
                    # choose a random source & destination
                    route_straight = np.random.randint(1, 5)
                    
                    if route_straight == 1:
                        print('    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)
                    elif route_straight == 2:
                        print('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)
                    elif route_straight == 3:
                        print('    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)
                    else:
                        print('    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)
                else:  # self._left_turn of the time, the car turns
                    route_turn = np.random.randint(1, 9)  # choose random source source & destination
                    if route_turn == 1:
                        print('    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)
                    elif route_turn == 2:
                        print('    <vehicle id="W_S_%i" type="standard_car" route="W_S" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)
                    elif route_turn == 3:
                        print('    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)
                    elif route_turn == 4:
                        print('    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)
                    elif route_turn == 5:
                        print('    <vehicle id="E_N_%i" type="standard_car" route="E_N" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)
                    elif route_turn == 6:
                        print('    <vehicle id="E_S_%i" type="standard_car" route="E_S" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)
                    elif route_turn == 7:
                        print('    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)
                    elif route_turn == 8:
                        print('    <vehicle id="S_E_%i" type="standard_car" route="S_E" depart="%s" '
                              'departLane="random" departSpeed="%f" />' % (car_counter, step, departspeed), file=routes)

            print("</routes>", file=routes)
