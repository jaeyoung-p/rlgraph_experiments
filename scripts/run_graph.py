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
import os

from aim import Run, Distribution, Figure


def run(cfg, simulator):

    experiment_name = cfg.dag.type + "_" + "sweep"

    run = Run(
        experiment=experiment_name,
        log_system_params=True,
    )
    run.add_tag("sweep")
    run.add_tag(cfg.dag.type)
    run["config"] = cfg

    current_directory = os.getcwd()
    simulator.run()
    final_time = simulator.get_current_time()
    run.track(name="total_time", value=final_time)

    # save to file
    with open(os.path.join(current_directory, "run.txt"), "w") as f:
        f.write("Total time: " + str(final_time))

    print("Total time: ", final_time)
