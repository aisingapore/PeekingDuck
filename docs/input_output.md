# Input and Output Nodes

## `input` nodes
Input nodes reads from a data source and pushes it down a PeekingDuck pipeline

| Node name | Inputs | Outputs | Parameters |
| `input.live`| source: placeholder for nodes that does not require any inputs | img: a numpy array of raw pixel values, shape (width, height, 3)   | resolution: Resolution of image from livefeed  |
|             |        |       | input_source: param to indicate input source This could be URLs for rtsp feeds, or 0 for local webcam |
|             |        |       | mirror_image: returns a horizontally flipped image |
| `input.recorded`| source: placeholder for nodes that does not require any inputs | img: a numpy array of raw pixel values, shape (width, height, 3) | resolution: Resolution of image from livefeed  |
|             |        | end: boolean to indicate that the frame of the last file is completed | input_source: param to indicate input source This could be URLs for rtsp feeds, or 0 for local webcam |
|             |        | filename: name of the file being read      | mirror_image: returns a horizontally flipped image |
|             |        | fps: fps of video feed      | mirror_image: returns a horizontally flipped image |





## `output` nodes
Output nodes takes outputs from a PeekingDuck pipeline to write or display it.