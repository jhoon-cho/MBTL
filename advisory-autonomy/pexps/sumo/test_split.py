from pexps.exp import *
from pexps.env import *
from u import *

class TestEnv(Env):
    def def_sumo(self):
        c = self.c
        n = lambda x: 'node_%s' % x
        e = lambda x: 'edge_%s' % x
        r = lambda x: 'route_%s' % x
        node_xs = np.cumsum(c.edge_lengths)
        nodes = E('nodes',
            E('node', id=n(0), x=0, y=0), *(
                E('node', id=n(1 + i), x=x, y=0, type='zipper', radius=20) for i, x in enumerate(node_xs)
            ),
            E('node', id=n(4), x=node_xs[-1], y=100)
        )
        edges = E('edges', *(
                E('edge', **{'id': e(i), 'from': n(i), 'to': n(i + 1), 'spreadType': 'center',
                    'numLanes': n_lane,
                    'speed': c.speed_limit
                }) for i, n_lane in enumerate(c.n_lanes)
            ),
            E('edge', **{'id': e(3), 'from': n(2), 'to': n(4)})
        )
        connections = E('connections',
            E('connection', **{'from': e(0), 'to': e(1), 'fromLane': 0, 'toLane': 0}),
            E('connection', **{'from': e(1), 'to': e(2), 'fromLane': 1, 'toLane': 0}),
            E('connection', **{'from': e(1), 'to': e(3), 'fromLane': 1, 'toLane': 0}),
        )
        routes = E('routes',
            E('route', id=r(0), edges='edge_0 edge_1 edge_3'),
            E('route', id=r(1), edges='edge_0 edge_1 edge_2'),
            E('flow', **FLOW('human', type='human', route=r(0),
                departSpeed=10, departLane='random', vehsPerHour=600)),
            E('flow', **FLOW('rl', type='rl', route=r(1),
                    departSpeed=10, departLane='random', vehsPerHour=1000))
        )
        additional = E('additional',
            E('vType', id='human', **IDM, **LC2013),
            E('vType', id='rl', **IDM, **LC2013)
        )
        return super().def_sumo(nodes, edges, connections, routes, additional)

class Test(Main):
    def create_env(c):
        return TestEnv(c)

    @property
    def observation_space(c):
        return Box(low=0, high=1, shape=(1,), dtype=np.float32)

    @property
    def action_space(c):
        return Box(low=0, high=1, shape=(1,), dtype=np.float32)

if __name__ == '__main__':
    c = Test.from_args()
    c.setdefaults(
        horizon=1000,
        n_steps=300,
        step_save=50,

        av_frac=0.1,
        flow_rate=2300,
        sim_step=0.5,
        edge_lengths=[400, 100, 200],
        n_lanes=[1, 2, 4],

        render=False,
        lr=5e-5,

        speed_limit=23,
        lc_mode=LC_MODE.strategic
    )
    c.run()