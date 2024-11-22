import logging
from numbers import Number
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from misc_utils import from_torch


def lif(keep, *x):
    return x if keep else []


def normalize(x):
    return (x - x.mean()) / x.std()


class Algorithm:
    """
    RL algorithm interface
    """

    def __init__(self, config, optimizer, model, dist_class):
        self.config = config
        self.optimizer = optimizer
        self.model = model
        self.dist_class = dist_class

    def on_step_start(self, current_step):
        return {}

    def optimize(self, rollouts, current_step) -> Dict[str, Number]:
        raise NotImplementedError

    def value_loss(self, v_pred, ret, v_start=None, mask=None):
        mask = slice(None) if mask is None else mask
        unclipped = (v_pred - ret) ** 2
        if v_start is None or self.config.vclip is None:  # no value clipping
            return unclipped[mask].mean()
        clipped_value = v_start + (v_pred - v_start).clamp(-self.config.vclip, self.config.vclip)
        clipped = (clipped_value - ret) ** 2
        return torch.max(unclipped, clipped)[mask].mean()  # no gradient if larger

    def step_loss(self, loss):
        self.optimizer.zero_grad()
        if torch.isnan(loss):
            raise RuntimeError('Encountered nan loss during training')
        loss.backward()
        if self.config.normclip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.normclip)
        self.optimizer.step()


class TRPO(Algorithm):
    def __init__(self, config, optimizer, model, dist_class):
        super().__init__(config, optimizer, model, dist_class)

    def max_kl(self, current_step):
        return self.config.start_max_kl + \
               (self.config.end_max_kl - self.config.start_max_kl) * current_step / self.config.n_steps

    def on_step_start(self, current_step):
        return dict(max_kl=self.max_kl(current_step))

    def optimize(self, rollouts, current_step) -> Dict[str, Number]:
        metrics = {}

        batch = rollouts.filter('obs', 'policy', 'action', 'pg_obj', 'ret',
                                *lif(self.config.use_critic, 'value', 'adv'))
        (_, b), = batch.iter_minibatch(None, concat=self.config.batch_concat,
                                       device=self.config.device)  # to be consistent with PPO

        pg_obj = b.pg_obj if 'pg_obj' in b else b.adv if self.config.use_critic else b.ret

        if self.config.adv_norm:
            pg_obj = normalize(pg_obj)
        start_dist = self.dist_class(b.policy)
        start_logp = start_dist.logp(b.action)

        def surrogate(dist):
            p_ratio = (dist.logp(b.action) - start_logp).exp()
            return (pg_obj * p_ratio).mean()

        # objective is -policy_loss, actually here the p_ratio is just 1, but we care about the gradients
        pred = self.model(b.obs, value=True, policy=True)
        pred_dist = self.dist_class(pred['policy'])
        obj = surrogate(pred_dist)

        params = list(self.model.p_head.parameters())
        flat_start_params = parameters_to_vector(params).clone()
        grad_obj = parameters_to_vector(torch.autograd.grad(obj, params, retain_graph=True))

        # Make fisher product estimator
        kl = start_dist.kl(pred_dist).mean()  # kl is 0, but we care about the gradient
        grad_kl = parameters_to_vector(torch.autograd.grad(kl, params, create_graph=True))

        def fvp_fn(x):  # fisher vector product
            return parameters_to_vector(
                torch.autograd.grad(grad_kl @ x, params, retain_graph=True)
            ).detach() + x * self.config.damping

        def cg_solve(_fvp_fn, _b, nsteps):
            """
            Conjugate Gradients Algorithm
            Solves Hx = _b, where H is the Fisher matrix and _b is known
            """
            x = torch.zeros_like(_b)  # solution
            r = _b.clone()  # residual
            p = _b.clone()  # direction
            new_rnorm = r @ r
            for _ in range(nsteps):
                rnorm = new_rnorm
                fvp = _fvp_fn(p)
                alpha = rnorm / (p @ fvp)
                x += alpha * p
                r -= alpha * fvp
                new_rnorm = r @ r
                p = r + new_rnorm / rnorm * p
            return x

        step = cg_solve(fvp_fn, grad_obj, self.config.steps_cg)
        max_trpo_step = (2 * self.max_kl(current_step) / (step @ fvp_fn(step))).sqrt() * step

        with torch.no_grad():  # backtrack until we find best step
            improve_thresh = grad_obj @ max_trpo_step * self.config.accept_ratio
            step = max_trpo_step
            for i_scale in range(self.config.steps_backtrack):
                vector_to_parameters(flat_start_params + step, params)
                test_dist = self.dist_class(self.model(b.obs, policy=True)['policy'])
                test_obj = surrogate(test_dist)
                kl = start_dist.kl(test_dist).mean()
                if kl < self.max_kl(current_step) and test_obj - obj > improve_thresh:
                    break
                step /= 2
                improve_thresh /= 2
            else:
                vector_to_parameters(flat_start_params, params)
                test_dist = start_dist
                test_obj = obj
                kl = 0

        if self.config.use_critic:
            shared = getattr(self.model, 'shared', None)
            if shared is not None:
                assert len(shared) == 0, 'Value network and policy network cannot share weights'
            for i_gd in range(self.config.n_gds):
                batch_stats = []
                for idxs, mb in rollouts.iter_minibatch(self.config.n_minibatches,
                                                        concat=self.config.batch_concat,
                                                        device=self.config.device):
                    pred = self.model(mb.obs.float(), value=True)
                    value_mask = mb.obs.get('value_mask') if isinstance(mb.obs, dict) else None
                    value_loss = self.value_loss(pred['value'], mb.ret, mask=value_mask)
                    self.step_loss(value_loss)
                    batch_stats.append(dict(value_loss=from_torch(value_loss)))
                # TODO whether overriding the value at each step makes sense
                metrics['value_loss'] = pd.DataFrame(batch_stats).mean(axis=0)['value_loss'].item()
        entropy = test_dist.entropy().mean()
        # The item() is important because tensors might still be on the GPU
        metrics.update(dict(policy_loss=-obj.item(),
                            final_policy_loss=-test_obj.item(),
                            i_scale=i_scale,
                            kl=kl.item(),
                            entropy=entropy.item()))

        return metrics


class PPO(Algorithm):
    def __init__(self, config, optimizer, model, dist_class):
        super().__init__(config, optimizer, model, dist_class)
        self.klcoef = self.config.klcoef_init

    def on_step_start(self, current_step):
        stats = dict(klcoef=self.klcoef)
        if self.config.entcoef:
            stats['entcoef'] = self.entcoef(current_step)
        return stats

    def entcoef(self, current_step):
        return schedule(current_step, self.config.n_steps, self.config.entcoef, self.config.ent_schedule)

    def optimize(self, rollouts, current_step) -> Dict[str, any]:
        metrics = {}

        batch = rollouts.filter('obs', 'policy', 'action', 'pg_obj', 'ret',
                                *lif(self.config.use_critic, 'value', 'adv'))
        value_warmup = current_step < self.config.n_value_warmup

        stop_update = False
        for i_gd in range(self.config.n_gds):
            batch_stats = []
            for idxs, mb in batch.iter_minibatch(self.config.n_minibatches,
                                                 concat=self.config.batch_concat,
                                                 device=self.config.device):
                if not len(mb):
                    continue
                start_dist = self.dist_class(mb.policy)
                start_logp = start_dist.logp(mb.action)
                if 'pg_obj' not in batch:
                    mb.pg_obj = mb.adv if self.config.use_critic else mb.ret
                pred = self.model(mb.obs, value=True, policy=True)
                curr_dist = self.dist_class(pred['policy'])
                p_ratio = (curr_dist.logp(mb.action) - start_logp).exp()

                pg_obj = mb.pg_obj
                if self.config.adv_norm:
                    pg_obj = normalize(pg_obj)

                policy_loss = -torch.min(
                    pg_obj * p_ratio,
                    pg_obj * p_ratio.clamp(1 - self.config.pclip, 1 + self.config.pclip)  # no gradient if larger
                ).mean()

                kl = start_dist.kl(curr_dist).mean()
                entropy = curr_dist.entropy().mean()

                loss = policy_loss + self.klcoef * kl - self.entcoef(current_step) * entropy
                stats = dict(
                    policy_loss=policy_loss, kl=kl, entropy=entropy
                )

                if value_warmup:
                    loss = loss.detach()

                if self.config.use_critic:
                    value_mask = mb.obs.get('value_mask') if isinstance(mb.obs, dict) else None
                    value_loss = self.value_loss(pred['value'], mb.ret, v_start=mb.value, mask=value_mask)
                    loss += self.config.vcoef * value_loss
                    stats['value_loss'] = value_loss
                self.step_loss(loss)
                batch_stats.append(from_torch(stats))

                if kl >= self.config.kltarg:
                    logging.warning("Trust region may have been reached!")
                    stop_update = True
                    break

            # TODO whether overriding the value at each step makes sense
            metrics['value_loss'] = pd.DataFrame(batch_stats).mean(axis=0)['value_loss']

            if stop_update:
                break

        if self.config.klcoef_init:
            kl = from_torch(kl)
            if kl > 2 * self.config.kltarg:
                self.klcoef *= 1.5
            elif kl < 0.5 * self.config.kltarg:
                self.klcoef *= 0.5

        return metrics


def schedule(current_step, n_steps, coef, scheduler=None):
    if not scheduler and isinstance(coef, (float, int)):
        return coef
    frac = current_step / n_steps
    frac_left = 1 - frac
    if callable(coef):
        return coef(frac_left)
    elif scheduler == 'linear':
        return coef * frac_left
    elif scheduler == 'cosine':
        return coef * (np.cos(frac * np.pi) + 1) / 2