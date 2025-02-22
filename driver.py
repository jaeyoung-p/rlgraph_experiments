import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig
import os

from scripts.setup_system import setup_system
from scripts.setup_graph import setup_graph
from scripts.setup_simulator import setup_simulator
from scripts.run_graph import run


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # print hydra job name
    print(HydraConfig.get().job.name)

    _, simulator = setup_simulator(
        cfg,
        log=True,
    )
    run(cfg, simulator)


if __name__ == "__main__":
    my_app()
