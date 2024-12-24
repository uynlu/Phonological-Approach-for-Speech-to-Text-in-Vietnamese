from builders.registry import Registry
from executors.base_executor import BaseExecutor

META_EXECUTOR = Registry("EXECUTOR")

def build_task(config) -> BaseExecutor:
    task = META_EXECUTOR.get(config.executor)(config)

    return task
