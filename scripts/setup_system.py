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


def setup_fully_connected(cfg):
    mem = cfg.system.gpumem
    bandwidth = cfg.system.bandwidth
    latency = cfg.system.latency
    n_devices = 1 + cfg.system.ngpus
    devices = uniform_connected_devices(n_devices, mem, latency, bandwidth)
    return devices


def setup_system(cfg):
    if cfg.system.type == "fully_connected":
        return setup_fully_connected(cfg)
    else:
        raise ValueError("Unknown system type: {}".format(cfg.system.type))
