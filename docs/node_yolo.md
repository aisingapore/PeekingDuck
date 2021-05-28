# Node YOLO Glossary

## 1. Overview

“You Only Look Once,” or YOLO, is a family of convolutional neural netowrk models for object detection. The model is developed by Bochkovskiy et al and the original paper can be found [here](https://arxiv.org/pdf/2004.10934.pdf). The model is trained using the COCO dataset, which contains more than 200,000 images and [80 object categories](#Class-IDs-and-names).

### Input and outputs

For input, the model takes in an image that is stored as a 3 dimentional NumPy array.

For outputs, the model returns the bounding boxes, classes and confident scores of the detected objects. All these results are stored in an dictionary and can be accessed using the dictionary keys shown below. Representation for each of the output is discussed next.

| Name of output   | Dictionary key |
| ---------------- | -------------- |
| Bounding box     | bboxes         |
| Classes          | bbox_labels    |
| Confident scores | bbox_scores    |

```python

# Example retriving the bounding box cooridiante from the model outputs

outputs['bboxes'] = [[0.43037105 0.32492152 0.7751606  0.9728762 ]]
```

#### Bounding box - "bboxes"

A list of numpy arrays, where each numpy array contains the bounding box coordinates of an object detected:

- x1: top left x-coordinate
- y1: top left y-coordinate
- x2: bottom right x-coordinate
- y2: bottom right y-coordinate

The order of the bboxes corresponds to the order of "labels" and "scores".

```python
'bboxes' = array([[x1, y1, x2, y2]),
                  ...
                 [x1, y1, x2, y2]])

# Example

'bboxes': array([[0.27510923, 0.12325603, 0.8680737 , 1.        ]]
```

#### Classes - "bbox_labels"

A list of labels of the name of classes of object detected. The order of the labels corresponds to the order of "bboxes" and "scores".

```python
'bbox_labels' = [str, str, ..., str]

# Example

'bbox_labels': ['person']
```

#### Confident scores - "bbox_scores"

A tf tensor of the confidence scores for the objects predicted. The order of the scores corresponds to the order of "bboxes" and "labels". Note that the score is between 0 and 1.

```python
'bbox_scores' = <tf.Tensor: shape=(1,), dtype=float32, numpy=array([float, float, ..., float], dtype=float32)>

# Example

'bbox_scores':  <tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.34761652], dtype=float32)>
```

## 2. Configurable parameters

The YOLO node can be configured in [run_config.yml](run_config.yml). Here are the configurable parameters for the YOLO node.

### Model type

The YOLO node allows user to choose between YOLOv4-tiny (default) and YOLOv4. YOLOv4-tiny is the compressed version of YOLOv4. YOLOv4-tiny has a faster inference speed while YOLOv4 has a higher precision and accuracy. To use YOLOv4, follow the example shown below:

```yaml
nodes:
  - input.live
  - model.yolo:
      - model_type: v4
  - draw.bbox
  - output.screen
```

### Max Detections per class

The maximum number of detected instances for each class in an image is set at _50_ by default. The maximum number per class can be adjusted as shown below:

```yaml
nodes:
  - input.live
  - model.yolo:
      - max_output_size_per_class: 10 # Reduce the number of detections to 10 per class
  - draw.bbox
  - output.screen
```

### Max Total Detections

The maximum total number of detected instances in an image is set at _50_ by default. The maximum number of total detection can be adjusted as shown below:

```yaml
nodes:
  - input.live
  - model.yolo:
      - max_total_size: 100 # Increse the total number of detections to 100
  - draw.bbox
  - output.screen
```

### Intersection over Union (IoU) threshold

For each class, overlapping bounding boxes above this IoU threshold will be discarded. The default IoU threshold is set at _0.5_. The IoU threshold can be adjusted between _0_ and _1_. To adjust the IoU threshold:

```yaml
nodes:
  - input.live
  - model.yolo:
      - yolo_iou_threshold: 0.2 # Decrease the IoU threshold to 0.5
  - draw.bbox
  - output.screen
```

### Score threshold

The score threshold refers to the minimum confidence the model has on a detected object(box confidence score). Any bounding box with a confident score lesser than the specified score threshold will be discarded. The default score threshold is set at _0.2_. The score threshold can be adjusted between _0_ and _1_. To adjust the score threshold:

```yaml
nodes:
  - input.live
  - model.yolo:
      - yolo_score_threshold: 0.5 # Increase the score threshold to 0.5
  - draw.bbox
  - output.screen
```

### Objects being detected

The YOLO node by default detects only **human** ('person'). To configure the node for detection of other objects, identify the objects' corresponding class ID from the table shown below and provide these IDs in an array identical to the following format:

```yaml
nodes:
  - input.live
  - model.yolo:
      - detect_ids: [0, 4, 41] # To detect 'person', 'aeroplane' and 'cup'
  - draw.bbox
  - output.screen
```

#### Class IDs and names

| Class ID | Class name    | Class ID | Class name     | Class ID | Class name   | Class ID | Class name   |
| -------- | ------------- | -------- | -------------- | -------- | ------------ | -------- | ------------ |
| 0        | person        | 20       | elephant       | 40       | wine glass   | 60       | dining table |
| 1        | bicycle       | 21       | bear           | 41       | cup          | 61       | toilet       |
| 2        | car           | 22       | zebra          | 42       | fork         | 62       | tv           |
| 3        | motorcycle    | 23       | giraffe        | 43       | knife        | 63       | laptop       |
| 4        | aeroplane     | 24       | backpack       | 44       | spoon        | 64       | mouse        |
| 5        | bus           | 25       | umbrella       | 45       | bowl         | 65       | remote       |
| 6        | train         | 26       | handbag        | 46       | banana       | 66       | keyboard     |
| 7        | truck         | 27       | tie            | 47       | apple        | 67       | cell phone   |
| 8        | boat          | 28       | suitcase       | 48       | sandwich     | 68       | microwave    |
| 9        | traffic light | 29       | frisbee        | 49       | orange       | 69       | oven         |
| 10       | fire hydrant  | 30       | skis           | 50       | broccoli     | 70       | toaster      |
| 11       | stop sign     | 31       | snowboard      | 51       | carrot       | 71       | sink         |
| 12       | parking meter | 32       | sports ball    | 52       | hot dog      | 72       | refrigerator |
| 13       | bench         | 33       | kite           | 53       | pizza        | 73       | book         |
| 14       | bird          | 34       | baseball bat   | 54       | donut        | 74       | clock        |
| 15       | cat           | 35       | baseball glove | 55       | cake         | 75       | vase         |
| 16       | dog           | 36       | skateboard     | 56       | chair        | 76       | scissors     |
| 17       | horse         | 37       | surfboard      | 57       | couch        | 77       | teddy bear   |
| 18       | sheep         | 38       | tennis racket  | 58       | potted plant | 78       | hair drier   |
| 19       | cow           | 39       | bottle         | 59       | bed          | 79       | toothbrush   |
