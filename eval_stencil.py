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

assignments = np.load("assignments.npy")
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


def init_weights(m):
    """
    Initializes LayerNorm layers.
    """
    if isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.weight, 1.0)
        torch.nn.init.constant_(m.bias, 0.0)


def logits_to_actions(
    logits, action=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.distributions.Categorical(logits=logits)
    if action is None:
        action = probs.sample()
    return action, probs.log_prob(action), probs.entropy()


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


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def my_app(cfg: DictConfig) -> None:
    args = Args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    lr = args.learning_rate
    epochs = args.num_iterations
    graphs_per_epoch = args.graphs_per_update

    devices = setup_system(cfg)
    tasks, _data = setup_graph(cfg)
    # Initialize a dummy simulator to get the graph features
    # Only GPUs are used for task computations
    H, sim = initialize_simulator(
        cfg,
        tasks,
        _data,
        devices,
    )
    candidates = sim.get_mapping_candidates()
    local_graph = sim.observer.local_graph_features(candidates)
    h = TaskAssignmentNetDeviceOnly(cfg.system.ngpus, args.hidden_dim, local_graph)
    h.load_state_dict(
        torch.load(
            "/Users/jaeyoung/work/rlgraph_experiments/outputs/ppo_Stencil_(10x10)2025-02-06 03:33:54/checkpoint_epoch_1200.pth",
            map_location=torch.device("cpu"),
            weights_only=True,
        )
    )

    netmap = GreedyNetworkMapper(h)
    H.set_python_mapper(netmap)

    for i in range(50, 81):
        cfg.dag.stencil.permute_idx = i
        devices = setup_system(cfg)
        tasks, _data = setup_graph(cfg)
        H, sim = initialize_simulator(
            cfg,
            tasks,
            _data,
            devices,
        )
        H.set_python_mapper(netmap)
        env = H.copy(sim)
        done = False
        baseline = H.copy(sim)
        baseline.disable_python_mapper()
        a = H.get_new_c_mapper()
        baseline.set_c_mapper(a)
        baseline_done = baseline.run()
        eft_time = baseline.get_current_time()
        print(eft_time)
        # Run env to first mapping
        env.set_python_mapper(netmap)
        env.enable_python_mapper()
        env.run()
        model_time = env.get_current_time()
        accuracy = eft_time / model_time * 100.0
        print(f"{i}: {accuracy:.2f}%")


if __name__ == "__main__":
    my_app()
