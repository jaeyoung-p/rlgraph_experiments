import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra
import os
import torch
import torch_geometric
import torch_geometric.data
import torch_geometric.loader
from scripts.setup_system import setup_system
from scripts.setup_graph import setup_graph
from scripts.setup_simulator import initialize_simulator, setup_simulator
from task4feedback.fastsim.interface import (
    PythonMapper,
    Action,
    Simulator,
    SimulatorHandler,
)
from scripts.run_graph import run
from task4feedback.fastsim.models import TaskAssignmentNetDeviceOnly
import logging
from task4feedback.types import (
    DataID,
    DataInfo,
    TaskID,
    TaskInfo,
    TaskPlacementInfo,
    TaskRuntimeInfo,
    Device,
    Architecture,
)
from task4feedback.graphs import *
import wandb
from datetime import datetime

import numpy as np
import random
import math
from dataclasses import dataclass
import concurrent.futures

# Load assignments (if needed) and disable logging
logging.disable(logging.CRITICAL)


@dataclass
class Args:
    hidden_dim = 64
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1
    """the discount factor gamma"""
    gae_lambda: float = 1
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 25000
    """the number of iterations (computed in runtime)"""

    graphs_per_update: int = 10
    """the number of graphs to use for each update"""
    reward: str = "percent_improvement"
    load_model: bool = False


def init_weights(m):
    """
    Initializes LayerNorm layers.
    """
    if isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.weight, 1.0)
        torch.nn.init.constant_(m.bias, 0.0)


def logits_to_actions(logits, action=None) -> tuple:
    probs = torch.distributions.Categorical(logits=logits)
    if action is None:
        action = probs.sample()
    return action, probs.log_prob(action), probs.entropy()


class GreedyNetworkMapper(PythonMapper):
    def __init__(self, model):
        self.model = model

    def map_tasks(self, candidates: np.ndarray, simulator):
        data = simulator.observer.local_graph_features(candidates, k_hop=1)
        with torch.no_grad():
            d, v = self.model.forward(data)
            # Choose argmax of network output for device assignment
            dev_per_task = torch.argmax(d, dim=-1)
            action_list = []
            for i in range(len(candidates)):
                if dev_per_task.dim() == 0:
                    dev_task = dev_per_task.item()
                else:
                    dev_task = dev_per_task[i].item()
                a = Action(
                    candidates[i],
                    i,
                    dev_task,
                    0,
                    0,
                )
                action_list.append(a)
        return action_list


class RandomNetworkMapper(PythonMapper):
    def __init__(self, model):
        self.model = model

    def map_tasks(self, candidates: np.ndarray, simulator, output=None):
        data = simulator.observer.local_graph_features(candidates, k_hop=1)
        with torch.no_grad():
            self.model.eval()
            d, v = self.model.forward(data)
            self.model.train()
            # Sample from network output
            dev_per_task, dlogprob, _ = logits_to_actions(d)

            if output is not None:
                output["candidates"] = candidates
                output["state"] = data
                output["dlogprob"] = dlogprob
                output["value"] = v
                output["dactions"] = dev_per_task

            action_list = []
            for i in range(len(candidates)):
                a = Action(
                    candidates[i],
                    i,
                    dev_per_task,
                    0,
                    0,
                )
                action_list.append(a)
        return action_list

    def evaluate(self, obs, daction, paction):
        p, d, v = self.model.forward(obs)
        _, plogprob, pentropy = logits_to_actions(p, paction)
        _, dlogprob, dentropy = logits_to_actions(d, daction)
        return (p, plogprob, pentropy), (d, dlogprob, dentropy), v


def run_episode(e, h, sim, block_time, args):
    """
    Runs a single simulation episode using a pre-created simulator instance.
    """
    done = False
    # Initialize the simulator (e.g. for the first step)
    obs, immediate_reward, done, terminated, info = sim.step()
    episode_info = []

    while not done:
        candidates = sim.get_mapping_candidates()
        record = {}
        action_list = RandomNetworkMapper(h).map_tasks(candidates, sim, record)
        obs, immediate_reward, done, terminated, info = sim.step(action_list)
        record["done"] = done
        record["time"] = sim.get_current_time()
        episode_info.append(record)

        if done:
            baseline = block_time
            if args.reward == "percent_improvement":
                record["reward"] = 1 + (baseline - record["time"]) / baseline
            elif args.reward == "exp":
                record["reward"] = math.exp((baseline - record["time"]) / baseline) - 1
            elif args.reward == "better":
                record["reward"] = 1 if record["time"] < baseline else 0

            print(f"{record['reward']}, {record['time']}, Block:{block_time}")
            break
        else:
            record["reward"] = 0

    with torch.no_grad():
        final_reward = episode_info[-1]["reward"]
        for t in range(len(episode_info)):
            episode_info[t]["returns"] = final_reward
            # Use a default of 0 if no 'value' exists.
            episode_info[t]["advantage"] = final_reward - episode_info[t].get(
                "value", 0
            )
    return episode_info


def collect_batch(episodes, h, H, SIM, block_time, args, global_step=0):
    """
    Parallelized version of collect_batch.

    Pre-creates an array of simulator copies—one for each episode—to avoid repeatedly
    calling H.copy(SIM) inside each thread.
    """
    batch_info = []
    # Create an array of independent simulator copies.
    simulators = [H.copy(SIM) for _ in range(episodes)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=episodes) as executor:
        # Submit each pre-created simulator along with its index.
        futures = [
            executor.submit(run_episode, e, h, sim, block_time, args)
            for e, sim in enumerate(simulators)
        ]
        for future in concurrent.futures.as_completed(futures):
            episode_info = future.result()
            batch_info.extend(episode_info)
    return batch_info


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def my_app(cfg: DictConfig) -> None:
    run_name = f"10g_ppo_Stencil_(10x10)" + datetime.today().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    if not os.path.exists(f"outputs/{run_name}"):
        os.makedirs(f"outputs/{run_name}")

    args = Args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    wandb.init(
        project="Stencil Adversarial Data Placement Single",
        name=run_name,
        config={
            "env_id": args.env_id,
            "total_timesteps": args.total_timesteps,
            "learning_rate": args.learning_rate,
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "num_minibatches": args.num_minibatches,
            "update_epochs": args.update_epochs,
            "norm_adv": args.norm_adv,
            "clip_coef": args.clip_coef,
            "clip_vloss": args.clip_vloss,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "max_grad_norm": args.max_grad_norm,
            "target_kl": args.target_kl,
            "batch_size": args.batch_size,
            "minibatch_size": args.minibatch_size,
            "num_iterations": args.num_iterations,
            "graphs_per_update": args.graphs_per_update,
            "reward": args.reward,
            "devices": cfg.system.ngpus,
            "vcus": 1,
            "blocks": cfg.dag.stencil.width,
        },
    )

    lr = args.learning_rate
    epochs = args.num_iterations
    graphs_per_epoch = args.graphs_per_update
    cfg.mapper.type = "block"
    cfg.mapper.python = True
    devices = setup_system(cfg)
    tasks, _data = setup_graph(cfg)
    # Initialize a dummy simulator to get the graph features.
    H, SIM = initialize_simulator(
        cfg,
        tasks,
        _data,
        devices,
    )
    block_sim = H.copy(SIM)
    block_sim.run()
    block_time = block_sim.get_current_time()

    candidates = SIM.get_mapping_candidates()
    local_graph = SIM.observer.local_graph_features(candidates)
    h = TaskAssignmentNetDeviceOnly(cfg.system.ngpus, args.hidden_dim, local_graph)
    optimizer = torch.optim.Adam(h.parameters(), lr=lr)
    rnetmap = RandomNetworkMapper(h)
    H.set_python_mapper(rnetmap)

    h.apply(init_weights)
    # Reinitialize the simulator after model initialization.
    H, SIM = initialize_simulator(
        cfg,
        tasks,
        _data,
        devices,
    )

    def batch_update(batch_info, update_epoch, h, optimizer, global_step):
        n_obs = len(batch_info)
        dclipfracs = []
        state = []
        total_rewards = 0
        for i in range(n_obs):
            state.append(batch_info[i]["state"])
            state[i]["dlogprob"] = batch_info[i]["dlogprob"]
            state[i]["value"] = batch_info[i]["value"]
            state[i]["dactions"] = batch_info[i]["dactions"]
            state[i]["advantage"] = batch_info[i]["advantage"]
            state[i]["returns"] = batch_info[i]["returns"]
            total_rewards += batch_info[i]["returns"]

        for k in range(update_epoch):
            nbatches = args.num_minibatches
            batch_size = n_obs // nbatches
            loader = torch_geometric.loader.DataLoader(
                state, batch_size=batch_size, shuffle=True
            )

            for i, batch in enumerate(loader):
                batch: torch_geometric.data.Batch
                d, v = h(batch, batch["tasks"].batch)
                da, dlogprob, dentropy = logits_to_actions(
                    d, batch["dactions"].detach().view(-1)
                )
                dlogratio = dlogprob.view(-1) - batch["dlogprob"].detach().view(-1)
                dratio = dlogratio.exp()

                with torch.no_grad():
                    dold_approx_kl = (-dlogratio).mean()
                    dapprox_kl = ((dratio - 1) - dlogratio).mean()
                    dclipfracs += [
                        ((dratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = batch["advantage"].detach().view(-1)
                dpg_loss1 = mb_advantages * dratio.view(-1)
                dpg_loss2 = mb_advantages * torch.clamp(
                    dratio.view(-1), 1 - args.clip_coef, 1 + args.clip_coef
                )
                dpg_loss = torch.min(dpg_loss1, dpg_loss2).mean()

                newvalue = v.view(-1)
                v_loss_unclipped = (newvalue - batch["returns"].detach().view(-1)) ** 2
                v_clipped = batch["value"].detach().view(-1) + torch.clamp(
                    newvalue - batch["value"].detach().view(-1),
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - batch["returns"].detach().view(-1)) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = dentropy.mean()
                loss = (
                    -1 * dpg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(h.parameters(), args.max_grad_norm)
                optimizer.step()

                wandb.log(
                    {
                        "losses/value_loss": v_loss.item(),
                        "losses/entropy": entropy_loss.item(),
                        "losses/dentropy": dentropy.mean().item(),
                        "losses/dratio": dratio.mean().item(),
                        "losses/dpolicy_loss": dpg_loss.item(),
                        "losses/dold_approx_kl": dold_approx_kl.item(),
                        "losses/dapprox_kl": dapprox_kl.item(),
                        "losses/dclipfrac": np.mean(dclipfracs),
                    },
                    commit=False,
                )
        wandb.log({"episode_reward": total_rewards / n_obs})

    for epoch in range(args.num_iterations):
        print("Epoch: ", epoch)
        # Use the parallelized collect_batch function. Each episode uses a pre-created simulator.
        batch_info = collect_batch(
            graphs_per_epoch, h, H, SIM, block_time, args, global_step=epoch
        )
        batch_update(batch_info, args.update_epochs, h, optimizer, global_step=epoch)

        # --- Gradient Monitoring ---
        total_norm = 0.0
        for name, param in h.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm**2
        total_norm = total_norm**0.5

        # Save the model every 100 epochs.
        if (epoch + 1) % 100 == 0:
            torch.save(
                h.state_dict(),
                f"outputs/{run_name}/checkpoint_epoch_{epoch+1}.pth",
            )

    # Save the final model.
    torch.save(h.state_dict(), f"outputs/{run_name}/model.pth")
    wandb.finish()


if __name__ == "__main__":
    my_app()
