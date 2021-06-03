# Node Glossary

This is a glossary of all the nodes currently available in PeekingDuck. To learn more about the parameters and the nodes, click on the link to read the docs.



### `input`
Reads data from a given input
|                             |                                                 |
| --------------------------- | ----------------------------------------------- |
| <img width=326 />           | <img width=654 />                               |
|  `input.live`               | Reads a videofeed from a stream (e.g. webcam)   |
|  `input.recorded`           | Reads a video/image from a file                 |


### `model`
Deep Learning models

| <img width=326 />                      | <img width=654 />                       |
| -------------------------------------- | --------------------------------------- |
| [`model.yolo`](./models/yolo.md)       | Fast Object Detection model             |
| [`model.posenet`](./models/posenet.md) | Fast Pose Estimation model              |
|  `model.efficientdet`                  | Slower, accurate Object Detection Model |


### `heuristic`
Algorithms that performs calculations/heuristics on the outputs of `model`

| <img width=326 />                    | <img width=654 />                                                 |
| ------------------------------------ | ----------------------------------------------------------------- |
|  `heuristic.bbox_count`              | Counts the number of detected boxes                               |
|  `heuristic.bbox_to_3d_loc`          | Estimates the 3D coordinates of an object given a 2D boundingbox  |
|  `heuristic.bbox_to_btm_midpoint`    | Converts bounding boxes to a single point of reference            |
|  `heuristic.check_large_groups`      | Check if number of objects in a group exceed a threshold          |
|  `heuristic.check_nearby_objs`       | Check if detected objects are near each other                     |
|  `heuristic.group_nearby_objs`       | Assign objects in close proximity to groups                       |
|  `heuristic.keypoints_to_3d_loc`     | Estimates the 3D coordinates of a human given 2D pose coordinates |
|  `heuristic.zone_count`              | Counts the number of detected objects within a boundary           |



### `output`
Writes/displays the outputs of the pipeline

| <img width=326 />         | <img width=654 />                    |
| ------------------------- | ------------------------------------ |
|  `output.media_writer`    | Write the output image/video to file |
|  `output.screen`          | Display the outputs on your display  |



### `draw`
Draws results/outputs to an image

| <img width=326 />             | <img width=654 />                                 |
| ----------------------------- | ------------------------------------------------- |
|  `draw.bbox`                  | Draw bounding boxes over detected object          |
|  `draw.bbox_count`            | Displays the counts of detected objects           |
|  `draw.fps`                   | Displays the FPS of video                         |
|  `draw.group_bbox_and_tag`    | Draws detected groups and their tags              |
|  `draw.btm_midpoint`          | Draws the lowest middle point of a bounding box   |
|  `draw.tag`                   | Displays a tag on bounding box                    |
|  `draw.poses`                 | Draws keypoints on a detected pose                |
|  `draw.zone_count`            | Displays counts of detected objects within a zone |
|  `draw.zones`                 | Draws the 2D boundaries of a zone                 |
