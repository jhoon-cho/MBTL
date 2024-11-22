from pathlib import Path
from typing import Union, Optional, Tuple, NamedTuple, Dict, List

import torch
import torch.nn as nn

from RL.algorithms import PPO
from containers.task_context import TaskContext

# define emission neural network 
class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       # our neural net has one input and one output layer with tanh activations
       self.relu_activation = nn.Tanh()
       self.layer1 = torch.nn.Linear(5, 32) 
       self.layer2 = torch.nn.Linear(32, 1)

   def forward(self, x):
       # given an input x, this function makes the prediction
       x = self.relu_activation(self.layer1(x))    
       x = self.layer2(x)      
       return x

class Config(NamedTuple):
    """
    Main experiment config.
    Even if it is not an immutable object, it MUST NOT BE MODIFIED after experiment.run() is called

    The fields which do NOT have an influence on the results are prefixed by _ so that they are ignored by WandB
    """
    ### Must be defined everytime

    run_mode: str
    """ What to do, can be either 'train', 'single_eval' or 'full_eval' """
    task_context: TaskContext
    """ defines the environment, see the TaskContext class"""
    working_dir: Path
    """ where to retrieve and store artifacts """
    source_dir: Path
    """ where to retrieve and store artifacts """

    ### General
    episode_to_eval: Union[str, int, Path] = 'max'
    """ When evaluating, which episode to choose. Can be a number, the path to the checkpoint or 'max' for the latest"""
    n_steps: any = 5000
    """ number of training or eval steps """
    parallel_rollout: bool = False # True
    """ Parallelize rollouts during training or full eval. Uses Ray. """
    parallelization_size: int = 1
    """
    number of workers
    Also number of rollout per training epochs #TODO add separate parameter
    """
    baseline_run: bool = False              
    """ Disable action from RL agents to do run baseline"""

    ### Outputs and monitoring
    logging: bool = True
    """ Verbose logging """
    visualize_sumo: bool = False
    """ whether to use SUMO's GUI, doesn't always work very well with Ray"""
    step_save: int = 50
    """ save model's checkpoint every x step """
    enable_wandb: bool = False # True
    """ Weights and Biases (see README) """
    wandb_proj: str = 'nostop'
    """ WandB project """
    moves_output: Optional[Path] = None
    """ If specified exports vehicle trajectories in the given dir to be processed by MOVES to get accurate emissions"""
    trajectories_output: bool = False
    """ Generates trajectories_emissions.xml file (made for single evals) for time-space diagrams """
    full_eval_run_baselines: bool = True
    """ in full_eval run mode, run the baseline (penrate = 0) and computes improvements """

    ### Simulation settings
    moves_emissions_models: List[str] = []
    """ Which (if any) MOVES surrogate to use """
    moves_emissions_models_conditions: List[str] = []
    """ What is the condition for each MOVES surrogate """
    """ Can be either regualr or electric. If electric, the emission model is only used for non-electric 
    vehicles and electric vehicles have zero emissions"""
    report_uncontrolled_region_metrics: bool = False
    """ Reports metrics both for controlled and uncontrolled region """
    csv_output_custom_name: str = ''
    """ Name of the csv file to store the metrics """
    control_lane_change: bool = False
    """ Whether to control controlled vehicles Lane change """
    sim_step_duration: float = 0.5
    """ Duration of SUMO steps """
    warmup_steps: int = 50  # in multi lane cannot do warmup
    """
    Number of Warmup steps at the beginning of the episode
    where vehicles are not controlled and metrics not collected
    """

    ### Reward
    stop_penalty: Optional[float] = 35
    threshold: float = 1
    accel_penalty: Optional[float] = None
    emission_penalty: Optional[float] = 3
    lane_change_penalty: Optional[float] = None
    optim_lane_penalty: Optional[float] = None
    fleet_reward_ratio: float = 0
    fleet_stop_penalty: Optional[float] = 0

    ### alg
    alg: type = PPO
    horizon: int = 1500
    use_critic: bool = True
    normclip: Optional[float] = None
    adv_norm: bool = False
    batch_concat: bool = True
    n_gds: int = 10
    n_minibatches: int = 40

    # for PPO only
    pclip: float = 0.03
    vcoef: float = 1
    vclip: float = 3.0
    klcoef_init: float = 0.1
    kltarg: float = 0.02
    entcoef: float = 0.005
    n_value_warmup: float = 0
    ent_schedule: any = None

    # #  for TRPO only
    # steps_backtrack: int = 10
    # steps_cg: int = 10
    # damping: float = 0.5
    # accept_ratio: float = 0.1
    # start_max_kl: float = 0.1
    # end_max_kl: float = 0.01

    ### model
    layers: Tuple = (256, 'tanh', 256, 'tanh', 256, 'tanh', 256, 'tanh')
    weight_scale: str = 'default'
    weight_init: str = 'orthogonal'

    ### optimizer
    lr: float = 1e-4
    lr_schedule: any = None
    gamma: float = 0.99
    lam: float = 0.97
    opt: str = 'Adam'
    betas: Tuple[float, float] = (0.9, 0.999)
    l2: float = 0
    aclip: bool = True

    ### Other
    device: any = 'cuda' if torch.cuda.is_available() else 'cpu'

    # function to load the emisison model that will be used in the reward function
    def load_emission_model():
        device_e = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = Net().to(device_e)
        # load the trained model
        net.load_state_dict(torch.load('resources/for_reward/2019_31.pt', map_location=torch.device('cpu')))
        # set the network to evaluation mode
        net.eval()
        return net

    ### Emission Model
    emissions_model = load_emission_model()

    def update(self, config_fields: Dict[str, any]) -> "Config":
        """
        Returns a NEW config object with the given fields
        """
        return Config(**{**self._asdict(), **config_fields})