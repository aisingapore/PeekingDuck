# Input, Output and Draw Nodes
Note: Users should not be editing the inputs and outputs of these nodes. We recommend users to only change the parameters as required.


## `input` nodes
Input nodes reads from a data source and pushes it down a PeekingDuck pipeline

| Node name | Inputs | Outputs | Parameters |
| - | - | - | - |
| [`input.live`](../peekingduck/configs/input/live.yml)| *source*: placeholder for nodes that does not require any inputs | *img*: a numpy array of raw pixel values, shape (width, height, 3), BGR format | *resize*: param to change the input image size to output. Value is a dictionary of three keys: <i>do_resizing</i>, <i>width</i> and <i>height</i>. If <i>do_resizing</i> is set to true, node will resize the input image to the desired <i>width</i> and <i>height</i>. By default, <i>do_resizing</i> is set to false <br />  *input_source*: param to indicate input source This could be URLs for rtsp feeds, or 0 for local webcam <br /> *mirror_image*: returns a horizontally flipped image |
| [`input.recorded`](../peekingduck/configs/input/recorded.yml)| *source*: placeholder for nodes that does not require any inputs | *img*: a numpy array of raw pixel values, shape (width, height, 3), BGR format <br /> *end*: boolean to indicate that the frame of the last file is completed <br /> *filename*: name of the file being read <br /> *fps*: fps of video feed | *resize*: param to change the input image size to output. Value is a dictionary of three keys: <i>do_resizing</i>, <i>width</i> and <i>height</i>. If <i>do_resizing</i> is set to true, node will resize the input image to the desired <i>width</i> and <i>height</i>. By default, <i>do_resizing</i> is set to false <br /> *input_dir*: directory of video/images to process. Could be a single file or directory.  <br /> *mirror_image*: returns a horizontally flipped image |





## `output` nodes
Output nodes takes outputs from a PeekingDuck pipeline to write or display it.
| Node name | Inputs | Outputs | Parameters |
| - | - | - | - |
|[`output.media_writer`](../peekingduck/configs/output/media_writer.yml) | *img*: a numpy array of pixel values, shape (width, height, 3), BGR format <br /> *filename*: filename of source <br /> *fps*: fps of source video | *end*: placeholder to indicate end of the pipeline | *output_dir*: output directory for files to be written locally |
|[`output.screen`](../peekingduck/configs/output/screen.yml) | *img*: a numpy array of pixel values, shape (width, height, 3), BGR format | *end*: placeholder to indicate end of the pipeline |

## `draw` nodes
Draw nodes draw the results from a PeekingDuck pipeline over the original video or image.
| Node name | Inputs | Outputs | Parameters |
| - | - | - | - |
| [`draw.bbox`](../peekingduck/configs/draw/bbox.yml) |*img*: a numpy array of pixel values, shape (width, height, 3), BGR format <br /> *bboxes*: A N by 4 NumPy array, where N represents the number of detected bounding boxes and 4 represents the four coordinates of each bounding box. | *img*: a numpy array of pixel values, shape (width, height, 3), BGR format | *bbox_color*: A list representing the color of the bounding box to be drawn, in BGR format <br /> *bbox_thickness*: An integer representing the thickness of the bounding box |
| [`draw.poses`](../peekingduck/configs/draw/poses.yml) | *img*: a numpy array of pixel values, shape (width, height, 3), BGR format <br /> *keypoints*: A _N_ by _17_ by _2_ NumPy array where N represents the number of detected human figures, 17 represents the number of keypoints and 2 represents the two spatial coordinates of each keypoint. <br /> *keypoints_conns*: A _N_ by _17_ by _2<sub>1</sub>_ by _2<sub>2</sub>_ NumPy array where _N_ represents the number of detected human figures and 17 represents the number of keypoints. 2<sub>1</sub> represents the two connecting keypoints if both keypoints are detected while 2<sub>2</sub> represents the two spatial coordinates of each corresponding connecting keypoint. | *img*: a numpy array of pixel values, shape (width, height, 3), BGR format |*keypoint_dot_color*: A list representing the color of the keypoints to be drawn, in BGR format <br /> *keypoint_dot_radius*: An integer representing the radius of the keypoint dot to be drawn <br /> *keypoint_connect_color*: A list representing the color of the connections between keypoints to be drawn, in BGR format <br /> *keypoint_text_color*: A list representing the color of the text describing the keypoints to be drawn, in BGR format <br /> |
| [`draw.fps`](../peekingduck/configs/draw/fps.yml)| *img*: a numpy array of pixel values, shape (width, height, 3), BGR format | *img*: a numpy array of pixel values, shape (width, height, 3), BGR format| |