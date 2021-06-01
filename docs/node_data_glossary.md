# Node Data Glossary

## 1. Overview

How does data get passed from one node to another? In [pipeline.py](peekingduck/pipeline/pipeline.py), `self._data` is a dictionary that accumulates data as nodes are triggered. If you already understand how this mechanism works and want more details on specific key-value pairs of available nodes, see [section](#2-key-value-pairs).

Here's an example when the pipeline runs [`input.live`, `model.yolo`].

When we inspect `self._data` after passing through `input.live`:

```
self._data = {"img": array}
```

Following this, after passing through `model.yolo`, which takes in "img" and returns "bboxes":

```
self._data = {"img": array, "bboxes": list}
```

Before the pipeline is run, a check is done to ensure that the inputs and the outputs of the chosen nodes tally. For example, without `input.live`, the check would fail because `model.yolo` will not be able to access "img".

This check is done by checking the input and output of each node's config file. For example, this is the config file for `input.live`.

```
input: ["source"]   # to be updated
output: ["img"]

# add other configs
```

This is the config file for `model.yolo`.

```
input: ["img"]
output: ["bboxes"]

# add other configs
```

This is the config file for `model.posenet`.

```
input: ["img"]
output: ["bboxes", "keypoints", "keypoint_scores", "keypoint_conns"]

# add other configs
```

TO-DO: Add diagram showing flow of data for a use case.

## 2. Key Value Pairs

### "bboxes"

A numpy array (N, 4) containing the bounding box information of detected objects. N corresponds to the number of object detected, and each detected bounding box is represented as (x1, y1, x2, y2) where:

- x1: top left x-coordinate
- y1: top left y-coordinate
- x2: bottom right x-coordinate
- y2: bottom right y-coordinate

The order of the bboxes corresponds to the order of "labels" and "scores".

```
"bboxes":   np.array(np.array([x1, y1, x2, y2]),
                     ...
                     np.array([x1, y1, x2, y2]))
```

### "bbox_labels"

A numpy array of labels of the name of classes of object detected. The order of the labels corresponds to the order of "bboxes" and "scores".

```
"labels":   np.array(str, str, ..., str)

example:    np.array("person", "person", ...)
```

### "bbox_scores"

A numpy array of the confidence scores for the objects predicted. The order of the scores corresponds to the order of "bboxes" and "labels". Note that the score is between 0 and 1.

```
"scores": np.array(float, float, ..., float)

example:
"scores": np.array(0.847334, 0.7039472, 0.243511)
```

### "keypoints"

A list of N numpy arrays, where each numpy array (Kx2) contains the (x, y) coordinates of the detected pose. N and K corresponds to the number of detected poses and number of keypoints respectively. If the keypoint has a low confidence score, the coordinates would be "masked" and replaced by "-1." as shown below.

```
[array([[ 0.58670201,  0.47576586],
       [ 0.60951909,  0.44109605],
                  ...
       [-1.        , -1.        ],
       [-1.        , -1.        ]])]
```       

### "keypoint_scores"

A list of N numpy arrays, where each numpy array (Kx1) contains the keypoint scores of the detected pose. N and K corresponds to the number of detected poses and number of keypoints respectively.

### "keypoint_conns"

A list of N numpy arrays, where each numpy array contains the keypoint connections
between adjacent keypoint pairs if both keypoints are detected.

### "end"

A boolean that is `True` if the end of a recorded video file is reached, and `False` if otherwise.

### "filename"

A string representing the filename of a recorded video/image that is being read.

### "fps"

A floating point number representing the FPS (Frames Per Second) that a recorded video was filmed at. This is usually between 25 - 30 FPS for most cameras.

### "img"

An image/ single video frame, in an array with shape (height, width, # colour channels). Note that we are using OpenCV and the colour channels are read in BGR order. The height and the width depends on the resolution of the image after preprocessing.

### "obj_3D_locs"

A list of numpy arrays, each containing the 3D coordinates of an object associated with a bounding box.

```
"obj_3D_locs": [np.array(x, y, z), ... , np.array(x, y, z)]
```

### "obj_groups"

A list of integers, each representing the allocated group number of an object associated with a bounding box.

```
# for example
"obj_groups": [1, 1, 1, 2, 3, 3, 3, 3, 3]
```

### "obj_tags"

A list of strings called tags, each tag associated with a bounding box. The order of the tags follow the order of "bboxes".

```
"obj_tags": [str, str, ... , str]

# for example
"obj_tags": ["TOO CLOSE!", "", ... , "TOO CLOSE!"]
```

### "zones"

A list of coordinate lists, each coordinate list is made of a list of tuples of (x, y) coordinates that demarks the points that form the boundaries of a zone. The order of zones follow the order of "zone_counts".
```
"zones":[[(int, int), (int, int), ...], [(int, int), (int, int), ...], ...,]

# for example
"zones": [[(0,0), (0, 500), (500, 500), (500, 0)], [(500, 500), (500, 1000), (1000, 1000), (1000, 500)]]
```

### "zone_count"

A list of integers that are counts of the number of chosen object (for example, people) detected in each specified zone. the order for the zone counts follow the order of "zones".

```
zone_count" [int, int, ..., int]

# for example
"zone_count": [1, 0, 5, 8, 4]
```
