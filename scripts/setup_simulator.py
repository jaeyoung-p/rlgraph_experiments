from task4feedback.fastsim.interface import (
    SimulatorHandler,
    uniform_connected_devices,
    TNoiseType,
    CMapperType,
    RoundRobinPythonMapper,
    Phase,
    PythonMapper,
    Action,
    start_logger,
    ExecutionState,
)
from typing import Optional, Self
from task4feedback.types import (
    DataID,
    TaskID,
    TaskPlacementInfo,
    TaskRuntimeInfo,
    Device,
    Architecture,
)
from task4feedback.graphs import StencilDataGraphConfig, StencilConfig, make_graph
from task4feedback.graphs.utilities import DataPlacer
from task4feedback.simulator.utility import parse_size
import numpy as np
from .setup_python_mapper import setup_python_mapper


def setup_task_noise(cfg):
    if cfg.env.task_noise == "None":
        return TNoiseType.NONE
    elif cfg.env.task_noise == "Lognormal":
        return TNoiseType.LOGNORMAL
    else:
        raise ValueError("Unknown noise type: {}".format(cfg.env.task_noise))


def setup_simulator(cfg, tasks, data, devices):
    start_logger()
    H = SimulatorHandler(
        tasks,
        data,
        devices,
        noise_type=setup_task_noise(cfg),
        seed=cfg.env.seed,
        cmapper_type=CMapperType.EFT_DEQUEUE,
    )
    simulator = H.create_simulator()
    simulator.initialize(use_data=cfg.env.use_data)
    simulator.randomize_durations()
    simulator.randomize_priorities()

    if cfg.mapper.python is True:
        print("Using Python mapper")
        python_mapper = setup_python_mapper(cfg, H.task_handle, data)
        simulator.set_python_mapper(python_mapper)
        simulator.enable_python_mapper()
    else:
        print("Using C++ mapper")
        simulator.disable_python_mapper()

    return simulator
