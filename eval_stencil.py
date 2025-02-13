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
    num_iterations: int = 1280
    """the number of iterations (computed in runtime)"""

    graphs_per_update: int = 10
    """the number of graphs to use for each update"""
    reward: str = "percent_improvement"
    load_model: bool = False


total_runs = 0
LI = 0


class GreedyNetworkMapper(PythonMapper):
    def __init__(self, model):
        self.model = model

    def map_tasks(self, candidates: np.ndarray[np.int32], simulator):
        data = simulator.observer.local_graph_features(candidates, k_hop=1)
        with torch.no_grad():
            d, v = self.model.forward(data)
            # Choose argmax of network output for priority and device assignment
            dev_per_task = torch.argmax(d, dim=-1)
            action_list = []
            for i in range(len(candidates)):
                # Check if p_per_task and dev_per_task are scalars
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


def logits_to_actions(
    logits, action=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.distributions.Categorical(logits=logits)
    if action is None:
        action = probs.sample()
    return action, probs.log_prob(action), probs.entropy()


class RandomNetworkMapper(PythonMapper):

    def __init__(self, model):
        self.model = model

    def map_tasks(self, candidates: np.ndarray[np.int32], simulator, output=None):
        data = simulator.observer.local_graph_features(candidates, k_hop=1)

        with torch.no_grad():
            self.model.eval()
            d, v = self.model.forward(data)
            self.model.train()

            # sample from network output
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


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def my_app(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    args = Args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    cfg.mapper.type = "block"
    cfg.mapper.python = True
    devices = setup_system(cfg)
    tasks, data = setup_graph(cfg)

    H_block, sim_block = initialize_simulator(
        cfg,
        tasks,
        data,
        devices,
    )
    sim_block.run()
    print(f"Block: {sim_block.get_current_time()}")

    cfg.mapper.type = "eft"
    cfg.mapper.python = False
    devices = setup_system(cfg)
    tasks, data = setup_graph(cfg)

    H_eft, sim_eft = initialize_simulator(
        cfg,
        tasks,
        data,
        devices,
    )
    sim_eft.run()
    print(f"EFT: {sim_eft.get_current_time()}")

    cfg.mapper.type = "block"
    cfg.mapper.python = True
    devices = setup_system(cfg)
    tasks, data = setup_graph(cfg)
    H_rl, sim_rl = initialize_simulator(
        cfg,
        tasks,
        data,
        devices,
    )
    candidates = sim_rl.get_mapping_candidates()
    local_graph = sim_rl.observer.local_graph_features(candidates)
    h = TaskAssignmentNetDeviceOnly(4, 64, local_graph)
    h.eval()
    netmap = GreedyNetworkMapper(h)
    H_rl.set_python_mapper(netmap)
    h.load_state_dict(
        torch.load(
            "/Users/jaeyoung/work/rlgraph_experiments/outputs/ppo_stencil_4x142025-02-13 00:17:39/checkpoint_epoch_3800.pth",
            map_location=torch.device("cpu"),
            weights_only=True,
        )
    )
    sim_rl = H_rl.copy(sim_rl)
    sim_rl.set_python_mapper(netmap)
    sim_rl.enable_python_mapper()
    sim_rl.run()
    print(f"RL: {sim_rl.get_current_time()}")

    for i in range(6, 9):
        accuracies = []
        for p in range(0, 24):
            cfg.dag.stencil.load_idx = i
            cfg.dag.stencil.permute_idx = p
            devices = setup_system(cfg)
            tasks, _data = setup_graph(cfg)
            H_rl, sim_rl = initialize_simulator(
                cfg,
                tasks,
                _data,
                devices,
            )
            H_rl.set_python_mapper(netmap)
            sim_rl = H_rl.copy(sim_rl)
            sim_rl.set_python_mapper(netmap)
            sim_rl.enable_python_mapper()
            sim_rl.run()
            rl_time = sim_rl.get_current_time()
            cfg.mapper.type = "eft"
            cfg.mapper.python = False
            H_eft, sim_eft = initialize_simulator(
                cfg,
                tasks,
                _data,
                devices,
            )
            sim_eft.run()
            eft_time = sim_eft.get_current_time()

            accuracy = eft_time / rl_time * 100.0
            accuracies.append(accuracy)
            print(f"EFT: {eft_time}, Model: {rl_time}, Accuracy: {accuracy:.2f}%")
        print(f"{i}: {average(accuracies):.2f}%")


if __name__ == "__main__":
    my_app()
