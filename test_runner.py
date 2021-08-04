from peekingduck.altrunner import Runner

if __name__ == "__main__":
    nodes = [
        "peekingduck.pipeline.nodes.input.live",
        "peekingduck.pipeline.nodes.model.yolo",
        "peekingduck.pipeline.nodes.draw.bbox",
        "peekingduck.pipeline.nodes.output.screen"
    ]
    runner = Runner(nodes)
    runner.run()
    print("End of Run")