from builders.executor_builder import build_executor
from configs.utils import get_config
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--config-file", type=str, required=True)
args = parser.parse_args()
config_file = args.config_file

if __name__ == "__main__":
    config = get_config(config_file)
    executor = build_executor(config)
    
    executor.start()
    executor.get_predictions()
    executor.logger.info("Completed!")
