# Node Data

## 1. Introduction

How does data get passed from one node to another? In [pipeline.py](peekingduck/pipeline/pipeline.py), self._data is a dictionary that aggregates data as nodes are traversed. If you already understand how this mechanism works and want more details on specific key-value pairs of available nodes, feel free to jump ahead to the next [section](#2-key-value-pairs).

For example, after passing through input.live:
```
self._data = {"img": array}
```
Following this, after passing through model.yolo, which takes in "img" and returns "bboxes":
```
self._data = {"img": array, "bboxes": list}
```

Before the pipeline is run, a check is done to ensure that the inputs and the outputs of the chosen nodes tally. For example, without input.live, the check would fail because model.yolo will not be able to access "img". 

This check is done by checking the input and output of each node's config file. For example, this is the config file for input.live.

```
input: ["source"]   # to be updated
output: ["img"]

# add other configs
```

This is the config file for model.yolo.

```
input: ["img"]
output: ["bboxes"]

# add other configs
```

TO-DO: Add diagram showing flow of data for a use case.

## 2. Key Value Pairs


### "bboxes"
A list of lists, where each list contains the bounding box coordinates of an object:
- x1: top left x-coordinate
- y1: top left y-coordinate
- x2: bottom right x-coordinate
- y2: bottom right y-coordinate
```
"bboxes":   [[x1, y1, x2, y2],
                ...
            [x1, y1, x2, y2]]
```

### "end"
A boolean that is `True` if the end of a recorded video file is reached, and `False` if otherwise. 


### "filename"
A string representing the filename of a recorded video/image that is being read.


### "fps"
A floating point number representing the FPS (Frames Per Second) that a recorded video was filmed at. This is usually between 25 - 30 FPS for most cameras.


### "img"
An image/ single video frame, in an array with shape (height, width, # colour channels). Note that we are using OpenCV and the colour channels are read in BGR order. The height and the width depends on the resolution of the image after preprocessing.



### "obj_3D_locs"

A list of dictionaries, each containing the index and 3D location of an object associated with a bounding box.
```
"obj_3D_locs": [{"idx": int, "3D_loc": np.array(x, y, z)},
                ...
                {"idx": int, "3D_loc": np.array(x, y, z)}]
```

### "obj_tags"

A list of dictionaries, each containing the index and tag an object associated with a bounding box.
```
"obj_3D_locs": [{"idx": int, "tag": str},
                ...
                {"idx": int, "tag": str}]
```



