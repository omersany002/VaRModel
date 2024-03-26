"""Main file to run the model"""

# importing packages
import yaml
import pandas as pd
from risk_model import VaRModel

# Extracting tickers from the fund summary
PATH = r'config\index-holdings-xly.xls'
df = pd.read_excel(PATH)
TICKERS = list(df[df.columns[0]])[1:]

# config file path
CONFIG_PATH = r'config\config.yaml'

# reading the configuration file
def load_config(config_path):
    """
    Load configuration settings from a YAML file.

    Parameters:
        config_path (str): Path to the configuration YAML file.

    Returns:
        dict: Loaded configuration data. Returns None if an error occurs.
    """

    # Try to open and load the YAML configuration file
    try:
        with open(config_path, 'r', encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    except FileNotFoundError:
        # Handle missing configuration file error
        return f"Error: Configuration file {config_path} not found."

    except yaml.YAMLError as e:
        # Handle YAML parsing errors
        return f"Error parsing configuration file: {e}"

def main(tickers, configuration_path):
    """Executes the script"""
    config = load_config(configuration_path)
    var = VaRModel(tickers, config)
    var.summary()

if __name__ == "__main__":
    main(TICKERS, CONFIG_PATH)
