import logging

from omegaconf import OmegaConf, DictConfig
import hydra

from hydra.core.hydra_config import HydraConfig

logger: logging.Logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info(f"runtime.output_dir{HydraConfig.get().runtime.output_dir}")


if __name__ == "__main__":
    main()
