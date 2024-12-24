from builders.registry import Registry

META_EXECUTOR = Registry("EXECUTOR")

def build_executor(config):
    task = META_EXECUTOR.get(config.executor)(config)

    return task
