import os
import peekingduck.runner as pkd


if __name__ == "__main__":
    RUN_PATH = os.path.join(os.getcwd(), 'PeekingDuck/run_config.yml')
    CUSTOM_NODE_PATH = os.path.join(os.getcwd(), 'PeekingDuck/custom_nodes')
    print(RUN_PATH)

    runner = pkd.Runner(RUN_PATH, CUSTOM_NODE_PATH)
    runner.run()