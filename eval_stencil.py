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
from scripts.setup_simulator import setup_simulator
from task4feedback.fastsim.interface import (
    PythonMapper,
    Action,
    Simulator,
    SimulatorHandler,
)
from scripts.run_graph import run
from scripts.visualize import save_boxplot
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


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def my_app(cfg: DictConfig) -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    randomness = 10
    torch.backends.cudnn.deterministic = True
    cfg.env.task_noise = "Lognormal"
    H_rl, sim_rl = setup_simulator(
        cfg,
    )
    candidates = sim_rl.get_mapping_candidates()
    local_graph = sim_rl.observer.local_graph_features(candidates)
    h = TaskAssignmentNetDeviceOnly(4, 64, local_graph)
    h.eval()
    h.load_state_dict(
        torch.load(
            "./saved_models/stencil_4x4_14steps_all_scenario_all_permute_rand(prior).pth",
            map_location=torch.device("cpu"),
            weights_only=True,
        )
    )
    netmap = GreedyNetworkMapper(h)
    results: Dict[str, List[float]] = {}
    runs = 0
    for i in range(0, 9):
        results[str(i)] = []
        for p in range(0, 24):
            cfg.dag.stencil.load_idx = i
            cfg.dag.stencil.permute_idx = p
            H_rl, sim_rl_base = setup_simulator(
                cfg, python_mapper=netmap, randomize_priorities=True
            )
            H_rl.set_python_mapper(netmap)
            for r in range(randomness):
                sim_rl = H_rl.copy(sim_rl_base)
                sim_rl.randomize_priorities()
                sim_eft = H_rl.copy(sim_rl)
                sim_rl.set_python_mapper(netmap)
                sim_rl.enable_python_mapper()
                sim_rl.run()
                rl_time = sim_rl.get_current_time()
                sim_eft.disable_python_mapper()
                sim_eft.run()
                eft_time = sim_eft.get_current_time()

                accuracy = eft_time / rl_time * 100.0 - 100.0
                results[str(i)].append(accuracy)
                runs += 1
                print(
                    f"{i}/{p} Speedup: {accuracy:.2f}%, {runs/(9*24*randomness)*100.0:.2f}% done"
                )

    save_boxplot(
        results,
        "Stencil Boundary Crossings Level",
        "Speedup (%)",
        12,
        f"stencil_accuracy_{cfg.dag.stencil.steps}.png",
    )


if __name__ == "__main__":
    my_app()
