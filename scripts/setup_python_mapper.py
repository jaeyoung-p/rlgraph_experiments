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


def device_to_idx(device: Device) -> int:
    if device.architecture == Architecture.CPU:
        return 0
    else:
        return device.device_id + 1


class StaticPythonMapper:
    def __init__(self, task_idx_to_device_id):
        self.mapping = task_idx_to_device_id

    def set_mapping(self, task_idx_to_device_id):
        self.mapping = task_idx_to_device_id

    def map_tasks(self, candidates: list[int], simulator) -> list[Action]:

        action_list = []
        for i, candidate in enumerate(candidates):
            device = device_to_idx(self.mapping(candidate))
            action_list.append(
                Action(
                    candidate,
                    i,
                    device,
                    0,
                    0,
                )
            )
        return action_list


def stencil_rowcyclic(cfg, tasks, data):

    def task_to_device(task_id: TaskID) -> Device:
        blocks = cfg.dag.stencil.width
        ngpus = cfg.system.ngpus

        device_id_i = int(task_id.task_idx[-2] % blocks)
        device_id_j = int(task_id.task_idx[-1] // blocks)

        idx = device_id_i + blocks * device_id_j
        device_id = idx % ngpus
        return Device(Architecture.GPU, device_id)

    def idx_to_device(task_idx: int) -> Device:
        return task_to_device(tasks.get_task_id(task_idx))

    return idx_to_device


def stencil_block(cfg, tasks, data):

    def task_to_device(task_id: TaskID) -> Device:
        blocks = cfg.dag.stencil.width
        ngpus = cfg.system.ngpus
        batch = ngpus // 2

        device_id_i = int(task_id.task_idx[-2] % batch)
        device_id_j = int(task_id.task_idx[-1] // batch)

        idx = device_id_i + batch * device_id_j
        device_id = idx % ngpus
        return Device(Architecture.GPU, device_id)

    def idx_to_device(task_idx: int) -> Device:
        return task_to_device(tasks.get_task_id(task_idx))

    return idx_to_device


def setup_python_mapper(cfg, tasks, data):
    if cfg.mapper.type == "block":
        return StaticPythonMapper(stencil_block(cfg, tasks, data))
    elif cfg.mapper.type == "rowcyclic":
        return StaticPythonMapper(stencil_rowcyclic(cfg, tasks, data))
    elif cfg.mapper.type == "random":
        raise NotImplementedError("Random mapper not implemented")
    elif cfg.mapper.type == "round_robin":
        mask = [0] + [1] * cfg.system.ngpus
        return RoundRobinPythonMapper(1 + cfg.system.ngpus, np.asarray(mask))
    else:
        raise ValueError("Unknown mapper type: {}".format(cfg.mapper.type))
