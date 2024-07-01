import hydra

from omegaconf import OmegaConf

@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def main(config: OmegaConf):
    config = OmegaConf.to_object(config)
    print(OmegaConf.to_yaml(config))

if __name__ == "__main__":
    main()
