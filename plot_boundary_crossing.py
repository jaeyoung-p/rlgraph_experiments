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

import imageio
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import math
import argparse

logging.disable(logging.CRITICAL)

total_runs = 0
LI = 0

# Define scale factor (e.g., 40x)
scale_factor = 40


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


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def my_app(cfg: DictConfig) -> None:
    run_mode = "RL"
    plot_composite = False
    random.seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)
    torch.manual_seed(cfg.env.seed)
    torch.backends.cudnn.deterministic = True
    print("Running simulation...")
    if run_mode == "RL":
        _H, _sim = setup_simulator(cfg)
        candidates = _sim.get_mapping_candidates()
        local_graph = _sim.observer.local_graph_features(candidates)
        h = TaskAssignmentNetDeviceOnly(4, 64, local_graph)
        h.eval()
        h.load_state_dict(
            torch.load(
                "/Users/jaeyoung/work/rlgraph_experiments/saved_models/stencil_4x4_14steps_all_scenario_all_permute_rand(prior).pth",
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
        sim.run()
    elif run_mode == "EFT":
        H, sim = setup_simulator(cfg)
        sim.run()
    else:
        raise ValueError("Unknown run mode: {}".format(run_mode))
    print("Simulation complete.")
    # Get the grid dimensions and number of steps from the config
    width = cfg.dag.stencil.width  # grid width (and height)
    num_steps = cfg.dag.stencil.steps  # number of steps

    # Colors for devices 0 to 3
    device_colors = {
        0: [255, 0, 0],  # Red
        1: [0, 255, 0],  # Green
        2: [0, 0, 255],  # Blue
        3: [255, 255, 0],  # Yellow
    }

    # Lists to store the snapshot frames and the boundary crossing counts per step
    frames = []
    boundary_counts = []

    # Iterate over each simulation step
    for step in range(num_steps):
        if plot_composite:
            snapshot = np.ones((width, width, 3), dtype=np.uint8) * 255
        device_matrix = np.empty((width, width), dtype=int)

        for x in range(width):
            for y in range(width):
                task_id = TaskID("T", (step, x, y), 0)
                task_index = H.task_handle.get_task_index(task_id)
                device = sim.get_mapping(task_index)
                device_matrix[x, y] = device
                if plot_composite:
                    snapshot[x, y] = device_colors.get(device, [0, 0, 0])

        # Count boundary crossings.
        boundary_cross = 0
        for x in range(width):
            for y in range(width):
                if x < width - 1 and device_matrix[x, y] != device_matrix[x + 1, y]:
                    boundary_cross += 1
                if y < width - 1 and device_matrix[x, y] != device_matrix[x, y + 1]:
                    boundary_cross += 1
        boundary_counts.append(boundary_cross)
        if plot_composite:
            # Upscale the snapshot using Pillow
            pil_img = Image.fromarray(snapshot)
            new_size = (width * scale_factor, width * scale_factor)
            pil_img = pil_img.resize(new_size, Image.NEAREST)

            # Draw grid lines for clarity
            draw = ImageDraw.Draw(pil_img)
            for i in range(1, width):
                x_pos = i * scale_factor
                draw.line([(x_pos, 0), (x_pos, new_size[1])], fill=(0, 0, 0), width=1)
            for j in range(1, width):
                y_pos = j * scale_factor
                draw.line([(0, y_pos), (new_size[0], y_pos)], fill=(0, 0, 0), width=1)

            # Add progress text showing current simulation step
            draw.text((10, 10), f"Step: {step}", fill=(0, 0, 0))

            frames.append(np.array(pil_img))

    # ---------------------------
    # Create a composite image showing each step's mapping result in a grid (5 columns) with margins
    # ---------------------------
    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)

    if plot_composite:
        cols = 5
        margin = 10  # space between images (in pixels)
        rows = math.ceil(len(frames) / cols)
        frame_width, frame_height = frames[0].shape[1], frames[0].shape[0]

        # Compute composite image size including margins:
        composite_width = cols * frame_width + (cols + 1) * margin
        composite_height = rows * frame_height + (rows + 1) * margin

        # Create a new blank composite image with white background.
        composite_img = Image.new(
            "RGB", (composite_width, composite_height), color=(255, 255, 255)
        )

        for idx, frame in enumerate(frames):
            row_idx = idx // cols
            col_idx = idx % cols
            frame_img = Image.fromarray(frame)
            # Calculate position with margin
            x_offset = margin + col_idx * (frame_width + margin)
            y_offset = margin + row_idx * (frame_height + margin)
            composite_img.paste(frame_img, (x_offset, y_offset))

        composite_path = os.path.join(
            output_dir,
            f"snapshot_{run_mode}_{width}x{width}_{num_steps}_min_{cfg.dag.stencil.permute_idx}.png",
        )
        composite_img.save(composite_path)
        print(f"Saved composite snapshot mapping image to {composite_path}")

    # ---------------------------
    # Plot the line graph for boundary crossings vs. step
    # ---------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(range(num_steps), boundary_counts, linestyle="-")
    plt.xlabel("Step")
    plt.ylabel("Number of Boundary Crossings")
    plt.title("Boundary Crossings per Step")
    plt.grid(True)
    plt.plot(range(num_steps), boundary_counts, linestyle="-")
    plt.axhline(
        y=2 * width,
        color="red",
        linewidth=3,
        label=f"Min Boundary Crossing ({2*width})",
    )
    plt.legend()
    line_plot_path = os.path.join(
        output_dir,
        f"line_{run_mode}_{width}x{width}_{num_steps}.png",
    )
    plt.savefig(line_plot_path)
    plt.close()
    print(f"Saved boundary crossing plot to {line_plot_path}")
    print(sim.get_current_time())


if __name__ == "__main__":
    my_app()
