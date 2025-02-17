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
import numpy as np
import random

# NEW: import imageio for saving GIFs
import imageio
from PIL import Image, ImageDraw, ImageFont

logging.disable(logging.CRITICAL)

total_runs = 0
LI = 0

# Define scale factor (e.g., 20x)
scale_factor = 40


def logits_to_actions(
    logits, action=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.distributions.Categorical(logits=logits)
    if action is None:
        action = probs.sample()
    return action, probs.log_prob(action), probs.entropy()


def init_weights(m):
    if isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.weight, 1.0)
        torch.nn.init.constant_(m.bias, 0.0)


class GreedyNetworkMapper(PythonMapper):
    def __init__(self, model):
        self.model = model

    def map_tasks(self, candidates: np.ndarray[np.int32], simulator) -> List[Action]:
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
    random.seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)
    torch.manual_seed(cfg.env.seed)
    torch.backends.cudnn.deterministic = True
    _H, _sim = setup_simulator(
        cfg,
    )
    candidates = _sim.get_mapping_candidates()
    local_graph = _sim.observer.local_graph_features(candidates)
    h = TaskAssignmentNetDeviceOnly(4, 64, local_graph)
    h.eval()
    h.load_state_dict(
        torch.load(
            "/Users/jaeyoung/work/rlgraph_experiments/checkpoint_epoch_3700.pth",
            map_location=torch.device("cpu"),
            weights_only=True,
        )
    )
    netmap = GreedyNetworkMapper(h)

    H, sim = setup_simulator(
        cfg,
        python_mapper=netmap,
        # log=True,
    )
    sim = H.copy(sim)
    sim.set_python_mapper(netmap)
    sim.enable_python_mapper()

    # ---------------------------
    # Visualization setup
    # ---------------------------
    # Define colors for each device (0 to 3)
    device_colors = {
        0: [255, 0, 0],  # Red
        1: [0, 255, 0],  # Green
        2: [0, 0, 255],  # Blue
        3: [255, 255, 0],  # Yellow
    }
    width = cfg.dag.stencil.width  # width of the matrix
    # We'll use a dict to store the latest mapping for each pixel
    # key: (row, col), value: (step, device)
    mapping = {}

    # List to store image frames (each frame is a numpy array)
    frames = []

    # ---------------------------
    # Simulation loop with visualization
    # ---------------------------
    obs, immediate_reward, done, terminated, info = sim.step()

    while not done:
        candidates = sim.get_mapping_candidates()
        action_list = netmap.map_tasks(candidates, sim)
        obs, immediate_reward, done, terminated, info = sim.step(action_list)

        # Get task information from the first action (as described)
        task_info = H.task_handle.get_task_id(action_list[0].taskid)
        step, row, col = task_info.task_idx  # task_idx is a tuple (step, row, col)

        # Update mapping: if the pixel (row, col) has not been mapped yet or
        # if the new mapping is from a different step, overwrite it.
        if (row, col) not in mapping or mapping[(row, col)][0] != step:
            mapping[(row, col)] = (step, action_list[0].device)

        # Create an image frame: start with a white background.
        frame = np.ones((width, width, 3), dtype=np.uint8) * 255
        # Color the pixels according to the mapping.
        for (r, c), (s, dev) in mapping.items():
            frame[r, c] = device_colors.get(dev, [0, 0, 0])

        # Upscale the frame using PIL
        pil_img = Image.fromarray(frame)
        resized_img = pil_img.resize(
            (width * scale_factor, width * scale_factor), Image.NEAREST
        )

        # Draw progress text on the resized image.
        draw = ImageDraw.Draw(resized_img)
        progress_text = f"Step: {step}"
        # Choose text position (for example, top-left corner) and color (black)
        text_position = (10, 10)
        text_color = (0, 0, 0)
        draw.text(text_position, progress_text, fill=text_color)

        # Convert back to numpy array (if needed) and append the frame.
        resized_frame = np.array(resized_img)
        frames.append(resized_frame)

    # print(
    #     f"{H.task_handle.get_task_id(action_list[0].taskid)} -> {action_list[0].device} {action_list[0].index}"
    # )

    # ---------------------------
    # Save the frames as a GIF
    # ---------------------------
    # Create an output directory if it doesn't exist
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
    )
    # Save the frames as a GIF (duration is time per frame in seconds)
    imageio.mimsave(output_path, frames, duration=0.5)
    print(f"Saved simulation visualization to {output_path}")


if __name__ == "__main__":
    my_app()
