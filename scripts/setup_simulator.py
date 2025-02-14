from task4feedback.fastsim.interface import (
    SimulatorHandler,
    Simulator,
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
from typing import Optional, Self, Mapping
from task4feedback.types import (
    TaskID,
    TaskInfo,
    DataID,
    DataInfo,
)
from task4feedback.graphs import StencilDataGraphConfig, StencilConfig, make_graph
from task4feedback.graphs.utilities import DataPlacer
from task4feedback.simulator.utility import parse_size
from task4feedback.fastsim.interface import Devices
import numpy as np
from .setup_python_mapper import setup_python_mapper
from .setup_system import setup_system
from .setup_graph import setup_graph


def setup_task_noise(cfg):
    if cfg.env.task_noise == "None":
        return TNoiseType.NONE
    elif cfg.env.task_noise == "Lognormal":
        return TNoiseType.LOGNORMAL
    else:
        raise ValueError("Unknown noise type: {}".format(cfg.env.task_noise))


def setup_simulator(
    cfg,
    tasks=None,
    data=None,
    devices=None,
    python_mapper=None,
    randomize_priorities=False,
    randomize_durations=False,
    log=False,
) -> tuple[SimulatorHandler, Simulator]:
    if cfg.env.task_noise == "None" and randomize_durations:
        raise ValueError("Cannot randomize durations without task noise")
    if cfg.env.task_noise == "None" and randomize_priorities:
        raise ValueError("Cannot randomize priorities without task noise")
    if tasks is None or data is None or devices is None:
        devices = setup_system(cfg)
        tasks, data = setup_graph(cfg)
    if log:
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

    if cfg.mapper.type == "block" and python_mapper is None:
        print("Using block mapper")
        simulator.initialize(use_data=cfg.env.use_data, use_transition_conditions=False)
    else:
        simulator.initialize(use_data=cfg.env.use_data, use_transition_conditions=True)

    if randomize_durations:
        simulator.randomize_durations()  # Not used if setup_task_noise is None
    if randomize_priorities:
        simulator.randomize_priorities()

    if python_mapper is not None:
        simulator.set_python_mapper(python_mapper)
        simulator.enable_python_mapper()
    elif cfg.mapper.python is True:
        python_mapper = setup_python_mapper(cfg, H.task_handle, data)
        simulator.set_python_mapper(python_mapper)
        simulator.enable_python_mapper()
    else:
        simulator.disable_python_mapper()
    return H, simulator
