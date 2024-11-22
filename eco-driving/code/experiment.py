import itertools
import random
from statistics import mean, stdev
from numbers import Number
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import ray
from ray import ObjectRef
from torch.nn import Module

from RL.models import FFN
from RL.utils import calc_adv, calc_adv_multi_agent, explained_variance, get_optimizer, \
    get_observation_space, get_action_space
from containers.config import Config
from containers.task_context import TaskContext
from containers.experiment_base import ExperimentBase, get_state
from RL.distributions import build_dist
from RL.algorithms import schedule
from env.environment import Env
from env.no_stop import NoStopRLEnv
from misc_utils.metrics import merge_rollout_metrics
from containers.nammed_arrays import NamedArrays
from misc_utils import to_torch, split, flatten, from_torch
from misc_utils.save_commit import save_commit
import wandb

import torch

import logging


class MainExperiment(ExperimentBase):
    """
    Manages the main training/eval loops
    """

    def __init__(self, config: Config):
        super().__init__(config)

        self.log_history = []
        self.env: Optional[Env] = None
        self.model = None
        self.current_step = 0
        self.alg = None
        self.n_rollouts_per_worker = 1
        self.opt = None
        self.rollout_workers = None
        self.observation_space = get_observation_space(self.config)
        self.action_space = get_action_space(self.config)
        self.dist_class = build_dist(self.action_space)

    @property
    def current_lr(self):
        return schedule(self.current_step, self.config.n_steps, self.config.lr, self.config.lr_schedule)

    def get_model(self) -> Module:
        """
        Returns the model according the config.
        """
        return FFN(self.config, self.observation_space, self.dist_class)

    def _init_logging(self) -> None:
        """
        Init the loggers.
        """
        logging.basicConfig(level=logging.DEBUG if self.config.logging else logging.INFO,
                            format="%(levelname)s %(asctime)s -- %(message)s",
                            handlers=[
                                logging.FileHandler(self.config.working_dir / 'log.txt'),
                                logging.StreamHandler()  # console output
                            ])

        save_commit(Path(''), self.config.working_dir)

        if self.config.enable_wandb:
            project = self.config.wandb_proj
            run = self.config.working_dir.name
            wandb.init(project=project,
                       name=run,
                       config=self.config._asdict(),
                       dir=self.config.working_dir,
                       resume=True)

    def log_metrics(self, metrics: Dict[str, any]) -> None:
        """
        Logs the given dict of metrics, can be called multiple times per step
        """
        if len(self.log_history) == self.current_step or len(self.log_history) == 0:
            # We fill in with empty values if the training was restarted
            # TODO save everything into a csv and recover it
            while len(self.log_history) <= self.current_step:
                self.log_history.append({})
        elif len(self.log_history) < self.current_step:
            raise AssertionError('log_metrics was not called at each step')

        self.log_history[self.current_step].update(metrics)

        for k, v in metrics.items():
            logging.debug(f'{self.current_step} | {k:30}: {v:.4f}')
            
        if self.config.run_mode == 'single_eval':
            df = self._results
            for k, v in metrics.items():
                if k not in df:
                    df[k] = np.nan
                df.loc[self.current_step, k] = v

        if self.config.enable_wandb:
            wandb.log(metrics, step=self.current_step)

    def save_checkpoint_if(self) -> None:
        """
        Saves a checkpoint of the current model if the conditions are met.
        """
        if self.current_step % self.config.step_save == 0:
            self.save_state(self.current_step, get_state(self.model, self.opt, self.current_step))

    def on_rollout_worker_start(self) -> None:
        """
        Initialises the objects needed to perform one or many rollouts.
        """
        self.env = NoStopRLEnv(self.config, self.observation_space)
        self.model = self.get_model()
        self.model.eval()

    def set_weights(self, weights: any) -> None:
        """
        Function used for loading the model parameters when using Ray.
        """
        self.model.load_state_dict(weights, strict=False)

    def collect_rollout_list(self,
                             eval_mode: bool = False,
                             force_single_thread: bool = False) -> Tuple[NamedArrays, Dict[str, any]]:
        """
        Collect a list of rollouts for the training step.

        Calls the ray workers if Ray is used.
        """
        if self.config.parallel_rollout and (not force_single_thread):
            weights_id = ray.put({k: v.cpu()
                                  for k, v in self.model.state_dict().items()})

            for w in self.rollout_workers:
                w.set_weights.remote(weights_id)

            rollouts_results = flatten(ray.get([
                w.rollouts_single_process.remote(worker_id=str(i), eval_seed=None if eval_mode is False else random.randint(0, 10000))
                for i, w in enumerate(self.rollout_workers)]))

        else:
            rollouts_results = self.rollouts_single_process(eval_seed=None if eval_mode is False else random.randint(0, 10000))

        # compute advantage estimate from the rollout
        processed_rollouts = [self.on_rollout_end(rollout[0])
                              for rollout in rollouts_results]

        # We take both the metrics from the env and from the on_rollout_end method,
        # and merge everything across rollouts
        metrics = merge_rollout_metrics([
            {**rollout[1], **processed_rollout[1]}
            for rollout in rollouts_results
            for processed_rollout in processed_rollouts])

        return NamedArrays.concat([rollout[0] for rollout in processed_rollouts], fn=flatten), metrics

    def rollouts_single_process(self, worker_id: str = '0', eval_seed: Optional[int] = None) -> List[Tuple[NamedArrays, Dict[str, any]]]:
        """
        Collect a rollout for the training step that can consist of multiple episodes.

        Is executed by ray workers when Ray is used.
        """
        if self.n_rollouts_per_worker > 1:
            return [self.rollout_episode(worker_id=worker_id)[:2]
                    for _ in range(self.n_rollouts_per_worker)]
        else:
            n_steps_total = 0
            rollouts = []
            while n_steps_total < self.config.horizon:
                rollout, metrics, steps = self.rollout_episode(worker_id=worker_id, eval_seed=eval_seed)
                rollouts.append((rollout, metrics))
                n_steps_total += steps
            return rollouts

    def rollout_episode(self, worker_id: str, task_context: Optional[TaskContext] = None, eval_seed: Optional[int] = None) -> \
            Tuple[NamedArrays, Dict[str, any], int]:
        """
        Collects the rollout for a single episode (a rollout might consist of multiple episodes).

        Performed by the ray workers if Ray is used.
        """
        task_context = self.config.task_context.sample_task() if task_context is None else task_context
        ids, obs = self.env.reset(worker_id if eval_seed is None else eval_seed, task_context)

        rollout = NamedArrays()
        rollout.append(id=ids, obs=obs)

        steps = 0
        done = False
        # rollout unfold
        while steps < self.config.horizon and not done:
            pred = from_torch(self.model(to_torch(rollout.obs[-1],
                                                  # when in ray, GPU is not available
                                                  device=self.config.device if torch.cuda.is_available() else 'cpu'),
                                         value=False,
                                         policy=True,
                                         argmax=eval_seed is not None))

            rollout.append(**pred)
            ids, obs, rewards, done = self.env.step(ids=rollout.id[-1],
                                                    action=rollout.action[-1],
                                                    warmup=False)

            rollout.append(id=ids, obs=obs, reward=rewards)
            steps += 1

        if self.config.moves_output is not None:
            self.config.moves_output.mkdir(exist_ok=True)
            self.env.export_moves_csv(
                self.config.moves_output / f'moves_{task_context.compact_str()}__{worker_id}.csv'
            )

        return rollout, self.env.get_metrics(), steps

    def on_rollout_end(self, rollout: NamedArrays) -> Tuple[NamedArrays, Dict[str, Number]]:
        """
        Compute value, calculate advantage, log stats after multiple rollouts (that can be performed in parallel).

        Performed by the main process (i.e. not Ray workers).
        """
        step_id = rollout.pop('id', None)
        multi_agent = step_id is not None

        step_obs = rollout.obs[:-1]
        assert len(step_obs) == len(rollout.reward)

        value_ = None
        if self.config.use_critic:
            (_, mb_), = rollout.filter('obs').iter_minibatch(concat=multi_agent, device=self.config.device)
            value_ = from_torch(self.model(mb_.obs, value=True)['value'].view(-1))

        if multi_agent:
            step_n = [len(x) for x in rollout.reward]
            reward = (np.concatenate(rollout.reward)).flatten()

            ret, adv = calc_adv_multi_agent(
                np.concatenate(step_id),
                reward,
                self.config.gamma,
                value_=value_,
                lam=self.config.lam)

            rollout.update(obs=step_obs, ret=split(ret, step_n))
            if self.config.use_critic:
                rollout.update(value=split(value_[:len(ret)], step_n), adv=split(adv, step_n))
        else:
            reward = rollout.reward
            ret, adv = calc_adv(reward, self.config.gamma, value_, self.config.lam)
            rollout.update(obs=step_obs, ret=ret)
            if self.config.use_critic:
                rollout.update(value=value_[:len(ret)], adv=adv)

        metrics = {
            'reward_avg': np.mean(reward),
            'ret_mean': np.mean(ret),
        }

        if self.config.use_critic:
            metrics['value_mean'] = np.mean(value_)
            metrics['adv_mean'] = np.mean(adv)
            metrics['explained_variance'] = explained_variance(value_[:len(ret)], ret)

        return rollout, metrics

    def init_workers(self) -> None:
        """
        Init the rollout workers (from the main thread).
        """
        if self.config.parallel_rollout:
            n_workers = self.config.parallelization_size
            self.n_rollouts_per_worker = self.config.parallelization_size // n_workers

            ray.init(num_cpus=n_workers, include_dashboard=False)
            remote_main = ray.remote(type(self))

            # can be done safely because the config is immutable
            rollout_workers_config = self.config

            self.rollout_workers: List[ObjectRef[MainExperiment]] = [
                remote_main.remote(config=rollout_workers_config)
                for _ in range(n_workers)]

            ray.get([w.on_rollout_worker_start.remote() for w in self.rollout_workers])

        else:
            n_workers = 1
            self.n_rollouts_per_worker = self.config.parallelization_size
            self.on_rollout_worker_start()

        assert n_workers * self.n_rollouts_per_worker == self.config.parallelization_size

    def train(self) -> None:
        """
        Main training loop
        """
        self.init_workers()

        self.model = self.get_model()
        self.opt = get_optimizer(self.config, self.current_lr, self.model)
        self.alg = self.config.alg(config=self.config,
                                   optimizer=self.opt,
                                   model=self.model,
                                   dist_class=self.dist_class)

        self.model.train()
        self.model.to(self.config.device)

        self.current_step = self.set_state(self.model, opt=self.opt, step='max')

        while self.current_step < self.config.n_steps:
            for g in self.opt.param_groups:
                g['lr'] = float(self.current_lr)

            with torch.no_grad():
                rollouts, rollout_metrics = self.collect_rollout_list()

            self.log_metrics(rollout_metrics)

            if len(rollouts.obs):
                opt_metrics = self.alg.optimize(rollouts, self.current_step)
                self.log_metrics(opt_metrics)

            self.log_metrics({'steps_trained': self.current_step})
            self.current_step += 1
            self.save_checkpoint_if()
            logging.debug('--------------------------')

        if hasattr(self.env, 'close'):
            self.env.close()

    def eval(self) -> None:
        """
        Main evaluation loop
        """

        self.model = self.get_model()

        self.model.eval()
        
        self._results = pd.DataFrame(index=pd.Series(name='step'))

        kwargs = {'step': self.config.episode_to_eval}
        step = self.set_state(self.model, opt=None, path=self.config.source_dir, **kwargs)

        logging.warning(f'Loaded model from step {step}')

        for _ in range(self.config.n_steps):
            # We reset the env each time in case the penrate changes
            self.env = NoStopRLEnv(self.config, self.observation_space)

            _, metrics = self.collect_rollout_list(eval_mode=True, force_single_thread=True)
            self.log_metrics(metrics)
            
            self._results.to_csv(self.config.working_dir / 'eval_result.csv')

            self.current_step += 1

            if hasattr(self.env, 'close'):
                self.env.close()

    def full_eval(self) -> None:
        """
        Evaluation that eval all possible tasks and reports results compared to IDM driving.

        Does not report result to WandB because WandB is poorly adapted to this kind of data.
        """
        self.init_workers()
        self.model = self.get_model()
        self.model.eval()
        kwargs = {'step': self.config.episode_to_eval}
        self.set_state(self.model, opt=None, **kwargs)
        self.model.to(self.config.device)

        n_workers = self.config.parallelization_size
        repeats = self.config.n_steps

        tasks: List[TaskContext] = self.config.task_context.list_tasks(add_0_penrate=self.config.full_eval_run_baselines)
        jobs = list(itertools.chain.from_iterable(itertools.repeat(x, repeats)
                                                  for x in tasks))

        logging.info(f'Evaluating the model on {len(jobs)} tasks')

        jobs_results = []

        for i in range(0, len(jobs), n_workers):
            logging.info(f'{i} out of {len(jobs)} evaluated.')
            jobs_range = jobs[i: i + n_workers]

            if self.config.parallel_rollout:
                weights_id = ray.put({k: v.cpu()
                                      for k, v in self.model.state_dict().items()})

                for w in self.rollout_workers:
                    w.set_weights.remote(weights_id)

                rollouts_results = ray.get([
                    w[0].rollout_episode.remote(worker_id=j,
                                                task_context=w[1],
                                                eval_seed=((i+j) % repeats) if repeats > n_workers else j)
                    for j, w in enumerate(zip(self.rollout_workers, jobs_range))])
            else:
                rollouts_results = [self.rollout_episode(worker_id=0, task_context=job, eval_seed=(i+j) % repeats)
                                    for j, job in enumerate(jobs_range)]

            jobs_results.extend([
                rollout_result[1]
                for rollout_result in rollouts_results
            ])

        keys = {key for jr in jobs_results for key in jr}

        aggregated_results = [
            {key: mean(max([0],
                           [result[key]
                            for result in jobs_results[i*repeats: (i+1)*repeats]
                            if key in result]))
             for key in keys}
            for i, _ in enumerate(tasks)
        ]

        for i, _ in enumerate(tasks):
            for key in keys:
                    r = []
                    for result in jobs_results[i*repeats: (i+1)*repeats]:
                        if key in result:
                            r.append(result[key])
                    if len(r) > 1:
                        aggregated_results[i][key + '_std'] = stdev(r)
                    else:
                        aggregated_results[i][key + '_std'] = 0

        if self.config.full_eval_run_baselines:

            for task, result in zip(tasks, aggregated_results):
                if task.penetration_rate != 0:
                    baseline_task = task._replace(penetration_rate=0)
                    baseline_result = aggregated_results[tasks.index(baseline_task)]

                    result.update({
                        k + '_improvement': (
                            (v - baseline_result[k]) / baseline_result[k] if baseline_result[k] != 0 else 0)
                        for k, v in result.items()
                        if k in baseline_result
                    })

        df_result = pd.concat([pd.DataFrame({**k._asdict(), **v}, index=[0])
                                   for k, v in zip(tasks, aggregated_results)])
        df_result.to_csv(self.config.working_dir /
                         f'{self.config.csv_output_custom_name}_{self.config.working_dir.name}_{self.config.episode_to_eval}_results.csv')

        if self.config.full_eval_run_baselines:
            
            for i, task in enumerate(tasks):
                if not task.penetration_rate == 0:

                    def comparison_result(name):
                        return f'{name}: {aggregated_results[i][name + "_improvement"] * 100:.1f} % ({aggregated_results[i][name]:.4f}) '

                    logging.info(' '.join([f'{k1}={v1}' for k1, v1 in task._asdict().items()]) + ': '
                                 + comparison_result('vehicle_speed')
                                 + comparison_result('vehicle_fuel')
                                 + comparison_result('num_vehicle'))

    def run(self) -> None:
        """
        Runs training or evaluation based on the config.
        """
        self._init_logging()
        logging.info(self.config)

        if self.config.run_mode == 'single_eval':
            logging.info("Evaluation mode!!")
            self.eval()
        elif self.config.run_mode == 'full_eval':
            self.full_eval()
        else:
            logging.info("Training mode!!")
            self.train()