class Config:
    """Configuration management class"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'training.num_epochs')"""
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def __getattr__(self, name):
        return self.config.get(name)
