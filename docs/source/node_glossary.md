# Node Glossary

This is a glossary of all the nodes currently available in PeekingDuck. To learn more about the parameters and the nodes, click on the links below to read the docs.

This document and the nodes available will be constantly updated; for now, these are what we think are the most important ones for you!


### `input`
Reads data from a given input. More details can be found [here](./io_draw_nodes.md#input-nodes).
| <img width=326 /> | <img width=654 />                             |
| ----------------- | --------------------------------------------- |
|  `input.live`     | Reads a videofeed from a stream (e.g. webcam) |
|  `input.recorded` | Reads video/images from a directory           |


### `model`
Deep Learning models

| <img width=326 />                      | <img width=654 />                       |
| -------------------------------------- | --------------------------------------- |
| [`model.yolo`](./models/yolo.md)       | Fast Object Detection model             |
| [`model.posenet`](./models/posenet.md) | Fast Pose Estimation model              |
| [`model.efficientdet`](./models/efficientdet.md) | Slower, accurate Object Detection Model |


### `output`
Writes/displays the outputs of the pipeline. More details can be found [here](./io_draw_nodes.md#output-nodes).

| <img width=326 />     | <img width=654 />                    |
| --------------------- | ------------------------------------ |
| `output.media_writer` | Write the output image/video to file |
| `output.screen`       | Show the outputs on your display     |


### `draw`
Draws results/outputs to an image. More details on `draw.bbox`, `draw.poses` and `draw.fps` can be found [here](./io_draw_nodes.md#draw-nodes).

| <img width=326 />          | <img width=654 />                                 |
| -------------------------- | ------------------------------------------------- |
|  `draw.bbox`               |  Draw bounding boxes over detected object         |
|  `draw.bbox_count`         | Displays the counts of detected objects           |
|  `draw.fps`                | Displays the FPS of video                         |
|  `draw.group_bbox_and_tag` | Draws detected groups and their tags              |
|  `draw.btm_midpoint`       | Draws the lowest middle point of a bounding box   |
|  `draw.tag`                | Displays a tag on bounding box                    |
|  `draw.poses`              | Draws keypoints on a detected pose                |
|  `draw.zone_count`         | Displays counts of detected objects within a zone |
|  `draw.zones`              | Draws the 2D boundaries of a zone                 |


### `dabble`
Algorithms that performs calculations/heuristics on the outputs of `model`

| <img width=326 />                 | <img width=654 />                                                 |
| --------------------------------- | ----------------------------------------------------------------- |
|  `dabble.bbox_count`              | Counts the number of detected boxes                               |
|  `dabble.bbox_to_3d_loc`          | Estimates the 3D coordinates of an object given a 2D boundingbox  |
|  `dabble.bbox_to_btm_midpoint`    | Converts bounding boxes to a single point of reference            |
|  `dabble.check_large_groups`      | Check if number of objects in a group exceed a threshold          |
|  `dabble.check_nearby_objs`       | Check if detected objects are near each other                     |
|  `dabble.group_nearby_objs`       | Assign objects in close proximity to groups                       |
|  `dabble.keypoints_to_3d_loc`     | Estimates the 3D coordinates of a human given 2D pose coordinates |
|  `dabble.zone_count`              | Counts the number of detected objects within a boundary           |
