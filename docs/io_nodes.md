# Input and Output Nodes
For all Input and Output nodes, there is little users can do about its inputs and outputs. We recommend users to only change the parameters


## `input` nodes
Input nodes reads from a data source and pushes it down a PeekingDuck pipeline

| Node name | Inputs | Outputs | Parameters |
| - | - | - | - |
| [`input.live`](../peekingduck/configs/input/live.yml)| *source*: placeholder for nodes that does not require any inputs | *img*: a numpy array of raw pixel values, shape (width, height, 3), BGR format | *resolution*: Desired resolution of image from livefeed <br />  *input_source*: param to indicate input source This could be URLs for rtsp feeds, or 0 for local webcam <br /> *mirror_image*: returns a horizontally flipped image |
| [`input.recorded`](../peekingduck/configs/input/recorded.yml)| *source*: placeholder for nodes that does not require any inputs | *img*: a numpy array of raw pixel values, shape (width, height, 3), BGR format <br /> *end*: boolean to indicate that the frame of the last file is completed <br /> *filename*: name of the file being read <br /> *fps*: fps of video feed | *resolution*: Desired resolution of image from file <br /> *input_dir*: directory of video/images to process. Could be a single file or directory.  <br /> *mirror_image*: returns a horizontally flipped image |





## `output` nodes
Output nodes takes outputs from a PeekingDuck pipeline to write or display it.
| Node name | Inputs | Outputs | Parameters |
| - | - | - | - |
|[`output.media_writer`](../peekingduck/configs/output/media_writer.yml) | *img*: a numpy array of pixel values, shape (width, height, 3) <br /> *filename*: filename of source <br /> *fps*: fps of source video | *end*: placeholder to indicate end of the pipeline | *output_dir*: output directory for files to be written locally |
|[`output.screen`](../peekingduck/configs/output/screen.yml) | *img*: a numpy array of pixel values, shape (width, height, 3) | *end*: placeholder to indicate end of the pipeline |