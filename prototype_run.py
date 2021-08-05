from peekingduck.mp_runner import Runner

if __name__ == "__main__":
    # running with 3 processes
    # "special" runs bbox and screen as one process
    nodes = [
        "peekingduck.pipeline.nodes.input.live",
        "peekingduck.pipeline.nodes.model.yolo",
        "special",
    ]

    # running 4 processes
    nodes2 = [
        "peekingduck.pipeline.nodes.input.live",
        "peekingduck.pipeline.nodes.model.yolo",
        "peekingduck.pipeline.nodes.draw.bbox",
        "peekingduck.pipeline.nodes.output.screen",
    ]

    runner = Runner(nodes)
    runner.run()
    print("End of Run")
