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
from task4feedback.graphs import CholeskyDataGraphConfig, CholeskyConfig, make_graph
from task4feedback.graphs.utilities import DataPlacer
from task4feedback.simulator.utility import parse_size
import numpy as np


def setup_stencil_data(cfg):
    data_config = StencilDataGraphConfig()
    data_config.n_devices = cfg.system.ngpus
    data_config.dimensions = cfg.dag.stencil.dimension
    data_config.width = cfg.dag.stencil.width

    interior_size = cfg.dag.stencil.interior_size * 4 * cfg.dag.stencil.data_scale

    def compute_boundary_size():
        return (
            np.sqrt(interior_size)
            * 4
            * cfg.dag.stencil.boundary_scale
            * cfg.dag.stencil.data_scale
        )

    boundary_size = compute_boundary_size()

    print(
        f"Time to move interior data: {interior_size / cfg.system.bandwidth}, Size: {interior_size}"
    )
    print(
        f"Time to move boundary data: {boundary_size / cfg.system.bandwidth}, Size: {boundary_size}"
    )

    def sizes(data_id: DataID) -> int:
        return boundary_size if data_id.idx[1] == 1 else interior_size

    data_config.initial_sizes = sizes

    if cfg.dag.stencil.initial_data_placement == "cpu":
        data_config.initial_placement = lambda data_id: (Device(Architecture.CPU, -1),)
    elif cfg.dag.stencil.initial_data_placement == "rowcyclic":

        def initial_data_placement_rowcyclic(data_id: DataID):
            ngpus = cfg.system.ngpus
            blocks = cfg.dag.stencil.width

            device_id_i = int(data_id.idx[-2] % blocks)
            device_id_j = int(data_id.idx[-1] // blocks)
            idx = device_id_i + blocks * device_id_j
            return Device(Architecture.GPU, idx % ngpus)

        data_config.initial_placement = initial_data_placement_rowcyclic

    elif cfg.dag.stencil.initial_data_placement == "colcyclic":
        raise NotImplementedError("colcyclic not implemented")
    elif cfg.dag.stencil.initial_data_placement == "block":

        def initial_data_placement_blocked(data_id: DataID):
            ngpus = cfg.system.ngpus
            batch = ngpus // 2

            device_id_i = int(data_id.idx[-2] // batch)
            device_id_j = int(data_id.idx[-1] // batch)
            idx = device_id_i + batch * device_id_j
            return Device(Architecture.GPU, idx % ngpus)

        data_config.initial_placement = initial_data_placement_blocked

    else:
        raise ValueError(
            "Unknown initial data placement: {}".format(
                cfg.dag.stencil.initial_data_placement
            )
        )

    return data_config


def setup_stencil(cfg):

    def task_config(task_id: TaskID) -> TaskPlacementInfo:
        placement_info = TaskPlacementInfo()
        placement_info.add(
            (Device(Architecture.GPU, -1),),
            TaskRuntimeInfo(task_time=1000, device_fraction=1),
        )
        return placement_info

    data_config = setup_stencil_data(cfg)

    config = StencilConfig(
        steps=cfg.dag.stencil.steps,
        width=data_config.width,
        dimensions=data_config.dimensions,
        task_config=task_config,
    )
    tasks, data = make_graph(config, data_config=data_config)

    return tasks, data


def setup_cholesky_data(cfg):
    data_config = CholeskyDataGraphConfig()
    data_config.data_size = cfg.dag.cholesky.data_size

    if cfg.dag.cholesky.initial_data_placement == "cpu":
        data_config.initial_placement = lambda data_id: (Device(Architecture.CPU, -1),)
    elif cfg.dag.cholesky.initial_data_placement == "rowcyclic":
        raise NotImplementedError("rowcyclic not implemented")
    else:
        raise ValueError(
            "Unknown initial data placement: {}".format(
                cfg.dag.stencil.initial_data_placement
            )
        )

    return data_config


def setup_cholesky(cfg):

    def task_config(task_id: TaskID) -> TaskPlacementInfo:
        placement_info = TaskPlacementInfo()
        placement_info.add(
            (Device(Architecture.GPU, -1),),
            TaskRuntimeInfo(task_time=1000, device_fraction=1),
        )
        placement_info.add(
            (Device(Architecture.CPU, -1),),
            TaskRuntimeInfo(task_time=1000, device_fraction=1),
        )
        return placement_info

    data_config = setup_cholesky_data(cfg)

    config = CholeskyConfig(
        blocks=cfg.dag.cholesky.blocks,
        task_config=task_config,
    )
    tasks, data = make_graph(config, data_config=data_config)

    return tasks, data


def setup_graph(cfg):
    if cfg.dag.type == "stencil":
        return setup_stencil(cfg)
    elif cfg.dag.type == "cholesky":
        return setup_cholesky(cfg)
    else:
        raise ValueError("Unknown dag type: {}".format(cfg.dag.type))
