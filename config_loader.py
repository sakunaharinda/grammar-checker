import yaml

def get_configs(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config
