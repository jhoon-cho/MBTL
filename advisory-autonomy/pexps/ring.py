from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))  # path handling for supercloud

from pexps.exp import *
from pexps.env import *
from u import *

class RingEnv(Env):
    def def_sumo(self):
        c = self.c

        self.rl_speeds = []
        self.all_speeds = []
        self.actions = []
        self.lane_changes = c.get('lane_changes')
        self.num_attempts = 0
        self.traj = {}
        r = c.circumference / (2 * np.pi)
        nodes = E('nodes',
            E('node', id='bottom', x=0, y=-r),
            E('node', id='top', x=0, y=r),
        )

        get_shape = lambda start_angle, end_angle: ' '.join('%.5f,%.5f' % (r * np.cos(i), r * np.sin(i)) for i in np.linspace(start_angle, end_angle, 80))
        edges = E('edges',
            E('edge', **{'id': 'right', 'from': 'bottom', 'to': 'top', 'length': c.circumference / 2, 'shape': get_shape(-np.pi / 2, np.pi / 2), 'numLanes': c.n_lanes}),
            E('edge', **{'id': 'left', 'from': 'top', 'to': 'bottom', 'length': c.circumference / 2, 'shape': get_shape(np.pi / 2, np.pi * 3 / 2), 'numLanes': c.n_lanes}),
        )

        connections = E('connections',
            *[E('connection', **{'from': 'left', 'to': 'right', 'fromLane': i, 'toLane': i}) for i in range(c.n_lanes)],
            *[E('connection', **{'from': 'right', 'to': 'left', 'fromLane': i, 'toLane': i}) for i in range(c.n_lanes)],
        )

        additional = E('additional',
            E('vType', id='human', **{**IDM, **LC2013, **dict(accel=1, decel=1.5, minGap=2, sigma=c.sigma)}),
            E('vType', id='rl', **{**IDM, **LC2013, **dict(accel=1, decel=1.5, minGap=2, sigma=0)}),
            *build_closed_route(edges, c.n_veh, c.av, space=c.initial_space)
        )
        return super().def_sumo(nodes, edges, connections, additional)

    @property
    def stats(self):
        stats = {k: v for k, v in super().stats.items() if 'flow' not in k}
        stats['circumference'] = c.circumference
        return stats

    def step(self, action=None):
        c = self.c
        ts = self.ts
        max_speed = c.max_speed
        circ_max = max_dist = c.circumference_max
        circ_min = c.circumference_min
        rl_type = ts.types.rl

        for veh in self.ts.vehicles.values():
            if veh.id not in self.traj:
                self.traj[veh.id] = []
            self.traj[veh.id].append((veh.laneposition, veh.lane.id))
            # at very low probability, we kick the system by setting the non-rl vehicles to a very low speed
            if c.get('kick') and np.random.uniform() < c.get('kick'):
                print("we kicked the vehicle at time {}!! (vehicle id: {})".format(self._step, veh.id))
                ts.tc.vehicle.slowDown(veh.id, 0.1, 1)

        if not rl_type.vehicles:
            super().step()
            return c.observation_space.low, 0, False, 0

        rl = nexti(rl_type.vehicles)
        if action is not None: # action is None only right after reset
            ts.tc.vehicle.setMinGap(rl.id, 0) # Set the minGap to 0 after the warmup period so the vehicle doesn't crash during warmup
            if c.act_type in ['velocity', 'vel_discrete', 'handcraft']:
                vel, lc = (action, None) if not c.lc_av else action if c.lc_act_type == 'continuous' else (action['vel'], action['lc'])
                if isinstance(vel, np.ndarray): vel = vel.item()
            elif c.act_type == 'accel_discrete':
                accel, lc = (action, None) if not c.lc_av else action if c.lc_act_type == 'continuous' else (action['accel'], action['lc'])
                if isinstance(accel, np.ndarray): accel = accel.item()
            else:
                accel, lc = (action, None) if not c.lc_av else action if c.lc_act_type == 'continuous' else (action['accel'], action['lc'])
                if isinstance(accel, np.ndarray): accel = accel.item()
                if isinstance(accel, (float, np.floating)):
                    accel = (accel - c.low) / (1 - c.low)
            if isinstance(lc, np.ndarray): lc = lc.item()
            if isinstance(lc, (float, np.floating)):
                lc = bool(np.round((lc - c.low) / (1 - c.low)))

            if c.act_type == 'handcraft':
                vel = c.handcraft[0] if self._step < c.skip_stat_steps*0.3 + c.warmup_steps else c.handcraft[1]
            else:
                if c.get('handcraft'):
                    if c.act_type.startswith('accel'):
                        accel = float(rl.speed < c.handcraft)
                        lc = True
                    elif c.act_type == 'velocity':
                        vel = c.handcraft if self._step < c.skip_stat_steps*0.3 + c.warmup_steps else action
                    elif c.act_type == 'vel_discrete':
                        vel = c.handcraft *(c.n_actions - 1)/c.max_speed if self._step < c.skip_stat_steps*0.3 + c.warmup_steps else action
                    elif c.act_type == 'accel_discrete':
                        vel = c.handcraft *(c.n_actions - 1)/c.max_speed if self._step < c.skip_stat_steps*0.3 + c.warmup_steps else action
                    else:
                        accel = action
                    # print(self._step, vel)

            if c.act_type == 'accel':
                acc = (accel * 2 - 1) * (c.max_accel if accel > 0.5 else c.max_decel)  # [0,1] --> m/s^2
                if c.rl_sigma:  # does not trigger when set to 0.0
                    acc += np.random.normal(scale=c.rl_sigma)
                ts.accel(rl, acc)
                # self.actions.append(action[0])
                self.actions.append(accel)
            elif c.act_type in ['velocity', 'handcraft']:  # FIXME(cathywu): partial implementation
                if c.rl_sigma:
                    vel += np.random.normal(scale=c.rl_sigma)
                ts.tc.vehicle.slowDown(rl.id, min(max(0, vel),c.max_speed), 1)
                # ts.set_speed(rl, vel)
                self.actions.append(vel)
            elif c.act_type == 'vel_discrete':
                level = vel / (c.n_actions - 1)
                ts.tc.vehicle.slowDown(rl.id, level*c.max_speed, 1e-3)
                self.actions.append(level*c.max_speed)
            elif c.act_type == 'accel_discrete':
                level = accel/ (c.n_actions - 1)
                ts.tc.vehicle.slowDown(rl.id, level*c.max_speed, 1e-3)
                self.actions.append(level*c.max_speed)
            else:
                if c.rl_sigma:  # WARNING: this is in normalized units
                    accel = np.clip(accel + np.random.normal(scale=c.rl_sigma), 0, 1)
                if c.get('constant_rl_velocity'):
                    level = float(c.constant_rl_velocity)
                elif c.act_type == 'continuous':
                    level = accel
                elif c.act_type == 'discretize':
                    level = min(int(accel * c.n_actions), c.n_actions - 1) / (c.n_actions - 1)
                elif c.act_type == 'discrete':
                    level = accel / (c.n_actions - 1)
                self.actions.append(level)
                ts.set_max_speed(rl, max_speed * level)
            if c.n_lanes > 1:
                if c.symmetric:
                    if lc:
                        ts.lane_change(rl, -1 if rl.lane_index % 2 else +1)
                else:
                    ts.lane_change_to(rl, lc)

        rl_speed = [veh.speed for veh in self.ts.types.rl.vehicles]
        all_speed = [veh.speed for veh in self.ts.vehicles.values()]
        avg_speed = sum(all_speed) / len(all_speed)
        self.rl_speeds.append(rl_speed[0])
        self.all_speeds.append(avg_speed)

        super().step()

        if len(ts.new_arrived | ts.new_collided):
            print('Detected collision')
            return c.observation_space.low, -c.collision_penalty, True, None
        elif len(ts.vehicles) < c.n_veh:
            print('Bad initialization occurred, fix the initialization function')
            return c.observation_space.low, 0, True, None

        leader, dist = rl.leader()
        if c.n_lanes == 1:
            obs = [rl.speed / max_speed, leader.speed / max_speed, dist / max_dist]
            if c.circ_feature:
                obs.append((c.circumference - circ_min) / (circ_max - circ_min))
            if c.accel_feature:
                obs.append(0 if leader.prev_speed is None else (leader.speed - leader.speed) / max_speed)
            if c.get('lane_changes'):
                # whenever the headway in front of the RL vehicle is large (usually ∼10m), another human-driven car may appear, based on the headway and the current speed of the car.
                for veh in ts.types['human'].vehicles:
                    leader, headway = veh.leader()
                    if headway > 10 and leader.speed > 0.1 and np.random.uniform()>0.5:
                        veh_id = 'human_{}'.format(22+self.num_attempts)
                        ts.add(veh_id, leader.route, type=ts.types['human'], lane_index=0, speed=veh.follower()[0].speed, pos=veh.laneposition - 5.)
                        self.num_attempts += 1
                num_vehicle = len(ts.types['human'].vehicles)
                if num_vehicle>22 and np.random.uniform()>0.5:
                    # ts.remove('human_{}'.format(22+self.num_attempts))
                    try:
                        veh_rm_id = np.random.choice(list(ts.types['human'].vehicles))
                        ts.vehicles.pop('{}'.format(veh_rm_id))
                        ts.remove('{}'.format(veh_rm_id))
                        print("completely removed vehicle {}".format(veh_rm_id))
                    except: pass
            if c.get('lane_changes_mayuri'):
                vehtype = ts.types['human']
                for route in ['route_right', 'route_left']:
                # route = rl.route
                    # speeds = [0, rl.leader()[0].speed, rl.speed]
                    speeds = [0, rl.leader()[0].speed]
                    for s in speeds:
                        veh_id = 'human_{}'.format(22+self.num_attempts)
                        # print(veh_id, s, ts.routes[route])
                        ts.add(
                            veh_id,
                            route=ts.routes[route],
                            type=vehtype,
                            lane_index=0,
                            speed=s,
                            pos=rl.leader()[0].laneposition - 5.)
                        self.num_attempts += 1

        elif c.n_lanes == 2:
            lane = rl.lane
            follower, fdist = rl.follower()
            if c.symmetric:
                other_lane = rl.lane.left or rl.lane.right
                oleader, odist = other_lane.next_vehicle(rl.laneposition, route=rl.route)
                ofollower, ofdist = other_lane.prev_vehicle(rl.laneposition, route=rl.route)
                obs = np.concatenate([
                    np.array([rl.speed, leader.speed, oleader.speed, follower.speed, ofollower.speed]) / max_speed,
                    np.array([dist, odist, fdist, ofdist]) / max_dist
                ])
            else:
                obs = [rl.speed]
                for lane in rl.edge.lanes:
                    is_rl_lane = lane == rl.lane
                    if is_rl_lane:
                        obs.extend([is_rl_lane, dist, leader.speed, fdist, follower.speed])
                    else:
                        oleader, odist = lane.next_vehicle(rl.laneposition, route=rl.route)
                        ofollower, ofdist = lane.prev_vehicle(rl.laneposition, route=rl.route)
                        obs.extend([is_rl_lane, odist, oleader.speed, ofdist, ofollower.speed])
                obs = np.array(obs) / [max_speed, *([1, max_dist, max_speed, max_dist, max_speed] * 2)]
        else:
            obs = [rl.speed]
            follower, fdist = rl.follower()
            for lane in rl.edge.lanes:
                is_rl_lane = lane == rl.lane
                if is_rl_lane:
                    obs.extend([is_rl_lane, dist, leader.speed, fdist, follower.speed])
                else:
                    oleader, odist = lane.next_vehicle(rl.laneposition, route=rl.route)
                    ofollower, ofdist = lane.prev_vehicle(rl.laneposition, route=rl.route)
                    obs.extend([is_rl_lane, odist, oleader.speed, ofdist, ofollower.speed])
            obs = np.array(obs) / [max_speed, *([1, max_dist, max_speed, max_dist, max_speed] * 3)]
        obs = np.clip(obs, 0, 1) * (1 - c.low) + c.low

        reward = np.mean([v.speed for v in (ts.vehicles if c.global_reward else rl_type.vehicles)])

        if action is not None and c.get('meta_stable_penalty') and 0.45 < level < 0.47:
            reward -= c.meta_stable_penalty
        if action is not None and c.get('incorrect_penalty'):
            reward -= c.incorrect_penalty * np.abs(level - 0.46)
        if c.accel_penalty and hasattr(self, 'last_speed'):
            reward -= c.accel_penalty * np.abs(rl.speed - self.last_speed) / c.sim_step
        if action is not None and c.get('inexec_penalty'):
            if c.act_type in ['velocity', 'vel_discrete', 'handcraft']:
                reward -= c.inexec_penalty * np.abs(level - np.mean([v.speed for v in rl_type.vehicles])/c.max_speed) / c.sim_step
            else:
                reward = reward
        if action is not None and c.get('insocial_penalty'):
            if c.act_type in ['velocity', 'vel_discrete', 'handcraft']:
                diff_insocials = []
                for v in rl_type.vehicles:
                    leader, headway = v.leader()
                    v_now = v.speed
                    s_star = 0 if leader is None else IDM['minGap'] + max(0, v_now * IDM['tau'] + v_now * (v_now - leader.speed) / (2 * np.sqrt(c.max_accel * c.max_decel)))
                    a_idm = max(c.max_accel, c.max_accel * (1 - (v_now / (level * c.max_speed + 0.001)) ** IDM['delta'] - (s_star / (headway - leader.length + 0.001)) ** 2) + np.random.normal(0, IDM['sigma']))
                    diff_insocials.append(np.abs(a_idm - c.max_accel) if a_idm >= 0 else np.abs(a_idm + c.max_decel))
                reward -= c.insocial_penalty * (np.mean(diff_insocials)/c.max_accel/c.sim_step)
            else:
                reward = reward
        self.last_speed = rl.speed

        return obs.astype(np.float32), reward, False, None

class Ring(Main):
    def create_env(c):
        return NormEnv(c, RingEnv(c))

    @property
    def observation_space(self):
        return Box(low=0, high=1, shape=(c._n_obs,), dtype=np.float32)

    @property
    def action_space(c):
        c.setdefaults(lc_av=False)
        assert c.act_type in ['discretize', 'discrete', 'continuous', 'accel', 'velocity', 'accel_discrete', 'vel_discrete', 'handcraft']
        if c.act_type in ['discretize', 'continuous', 'accel']:
            if not c.lc_av or c.lc_act_type == 'continuous':
                return Box(low=c.low*c.act_coef, high=1*c.act_coef, shape=(1 + bool(c.lc_av),), dtype=np.float32)
            elif c.lc_act_type == 'discrete':
                return Namespace(accel=Box(low=c.low*c.act_coef, high=1*c.act_coef, shape=(1,), dtype=np.float32), lc=Discrete(c.lc_av))
        elif c.act_type in ['velocity','handcraft']:
            if not c.lc_av or c.lc_act_type == 'continuous':
                return Box(low=0, high=c.max_speed*c.act_coef, shape=(1 + bool(c.lc_av),), dtype=np.float32)
            elif c.lc_act_type == 'discrete':
                return Namespace(vel=Box(low=0, high=c.max_speed, shape=(1,), dtype=np.float32), lc=Discrete(c.lc_av))
        elif c.act_type in ['discrete', 'accel_discrete', 'vel_discrete']:
            return Discrete(c.n_actions)

if __name__ == '__main__':
    c = Ring.from_args(globals(), locals()).setdefaults(
        n_lanes=1,
        horizon=2000,
        warmup_steps=1000,
        skip_stat_steps=2000,
        sim_step=0.1,
        av=1,
        max_speed=10,
        max_accel=0.5,
        max_decel=0.5,
        circumference=250,
        circumference_max=300,
        circumference_min=200,
        initial_space='free',
        sigma=0.2,

        circ_feature=False,
        accel_feature=False,
        act_type='accel',
        lc_act_type='continuous',
        low=-1,
        global_reward=False,
        accel_penalty=0,
        collision_penalty=100,
        rl_sigma=0.0,
        constant_rl_velocity=None,  # speed level, not absolute speed
        meta_stable_penalty=0,
        incorrect_penalty=0,
        inexec_penalty = None,
        insocial_penalty = None,
        n_actions=10, # number of actions for discrete action space

        hc_param=1,
        hc_reward='average',
        act_coef=1,
        
        lane_changes=False,
        
        kick=False,

        n_steps=1000,
        gamma=0.999,
        alg=TRPO,
        norm_reward=True,
        center_reward=True,
        adv_norm=False,
        step_save=5,

        render=False,
    )
    if c.n_lanes == 1:
        c.setdefaults(n_veh=22, _n_obs=3 + c.circ_feature + c.accel_feature)
    elif c.n_lanes == 2:
        c.setdefaults(n_veh=44, lc_mode=LC_MODE.no_lat_collide, symmetric=True, lc_av=2)
        c._n_obs = (1 + 2 * 2 * 2) if c.symmetric else (1 + 2 * 5)
    elif c.n_lanes == 3:
        c.setdefaults(n_veh=66, lc_mode=LC_MODE.no_lat_collide, symmetric=False, lc_av=3, _n_obs=1 + 3 * (1 + 2 * 2))
    c.step_save = c.step_save or min(5, c.n_steps // 10)
    c.run()
