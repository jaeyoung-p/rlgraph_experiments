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
from typing import Optional, Self, Tuple, Mapping, Dict
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
from task4feedback.graphs import StencilDataGraphConfig, StencilConfig, make_graph
from task4feedback.graphs import CholeskyDataGraphConfig, CholeskyConfig, make_graph
from task4feedback.graphs.utilities import DataPlacer
from task4feedback.simulator.utility import parse_size
import numpy as np
import itertools


def setup_stencil_data(cfg):
    data_config = StencilDataGraphConfig()
    data_config.n_devices = cfg.system.ngpus
    data_config.dimensions = cfg.dag.stencil.dimension
    data_config.width = cfg.dag.stencil.width
    if cfg.dag.stencil.interior_comm != "None":
        interior_size = 1000 * (
            int(cfg.dag.stencil.interior_comm) * (cfg.system.bandwidth)
        )
    else:
        interior_size = cfg.dag.stencil.interior_size * 4 * cfg.dag.stencil.data_scale

    if cfg.dag.stencil.boundary_comm != "None":
        boundary_size = 1000 * (
            int(cfg.dag.stencil.boundary_comm) * (cfg.system.bandwidth)
        )
    else:

        def compute_boundary_size():
            return (
                np.sqrt(interior_size)
                * 4
                * cfg.dag.stencil.boundary_scale
                * cfg.dag.stencil.data_scale
            )

        boundary_size = compute_boundary_size()

    # print(
    #     f"Time to move interior data: {interior_size / cfg.system.bandwidth}, Size: {interior_size}"
    # )
    # print(
    #     f"Time to move boundary data: {boundary_size / cfg.system.bandwidth}, Size: {boundary_size}"
    # )

    def sizes(data_id: DataID) -> int:
        return boundary_size if data_id.idx[1] == 1 else interior_size

    data_config.initial_sizes = sizes

    if cfg.dag.stencil.initial_data_placement == "cpu":
        data_config.initial_placement = lambda data_id: (Device(Architecture.CPU, 0),)
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
            if ngpus == 3:
                ngpus = 4
            batch = data_config.width // (ngpus // 2)

            device_id_i = int(data_id.idx[-2] // batch)
            device_id_j = int(data_id.idx[-1] // batch)
            idx = device_id_i + (ngpus // 2) * device_id_j

            dev_id = idx % ngpus
            if cfg.system.ngpus == 4:
                return Device(Architecture.GPU, dev_id)
            elif dev_id == 0:
                return Device(Architecture.CPU, 0)
            else:
                return Device(Architecture.GPU, dev_id - 1)

        data_config.initial_placement = initial_data_placement_blocked
    elif cfg.dag.stencil.initial_data_placement == "load":
        if cfg.system.ngpus not in [3, 4]:
            raise ValueError(
                f"Load placement only supported for 3GPU+1CPU or 4 GPUs, got {cfg.system.ngpus}"
            )
        M = cfg.dag.stencil.width
        if M % 2 != 0:
            raise ValueError(
                "Load placement only supported for even width, got {}".format(M)
            )
        placement_info = np.load(cfg.dag.stencil.placement_file_location)
        if placement_info.shape != (1 + 2 * M * (M - 2) // 2, M, M):
            raise ValueError(
                f"Placement info shape mismatch, expected {(1+2*M*(M-2)//2, M, M)}, got {placement_info.shape}"
            )
        # End file sanity check

        all_combinations = [list(c) for c in itertools.permutations(range(4))]
        if cfg.dag.stencil.load_idx >= len(placement_info):
            raise ValueError(
                f"Load index out of bounds Max: {len(placement_info)-1}, got {cfg.dag.stencil.load_idx}"
            )
        elif cfg.dag.stencil.permute_idx >= len(all_combinations):
            raise ValueError(
                f"Permutation index out of bounds Max: {len(all_combinations)-1}, got {cfg.dag.stencil.permute_idx}"
            )

        # End user input sanity check
        def initial_data_placement_load(data_id: DataID):
            dev_id = all_combinations[cfg.dag.stencil.permute_idx][
                placement_info[cfg.dag.stencil.load_idx][data_id.idx[-2]][
                    data_id.idx[-1]
                ]
            ]
            if cfg.system.ngpus == 4:  # 4 GPU setting
                return Device(Architecture.GPU, dev_id)
            elif dev_id == 0:  # 1 CPU 3 GPU setting
                return Device(Architecture.CPU, 0)
            else:
                return Device(Architecture.GPU, dev_id - 1)

        data_config.initial_placement = initial_data_placement_load

    else:
        raise ValueError(
            "Unknown initial data placement: {}".format(
                cfg.dag.stencil.initial_data_placement
            )
        )

    return data_config


def setup_stencil(cfg) -> Tuple[Mapping[TaskID, TaskInfo], Dict[DataID, DataInfo]]:

    def task_config(task_id: TaskID) -> TaskPlacementInfo:
        placement_info = TaskPlacementInfo()
        if cfg.system.ngpus == 3:
            placement_info.add(
                (Device(Architecture.CPU, -1),),
                TaskRuntimeInfo(task_time=1000, device_fraction=1),
            )
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


def setup_graph(cfg) -> Tuple[Mapping[TaskID, TaskInfo], Dict[DataID, DataInfo]]:
    if cfg.dag.type == "stencil":
        return setup_stencil(cfg)
    elif cfg.dag.type == "cholesky":
        return setup_cholesky(cfg)
    else:
        raise ValueError("Unknown dag type: {}".format(cfg.dag.type))
