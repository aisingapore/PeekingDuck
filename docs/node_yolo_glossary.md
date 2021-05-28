# Node YOLO Glossary

## 1. Overview

“You Only Look Once,” or YOLO, is a family of convolutional neural netowrk models for object detection. The model is trained using the COCO dataset, which contains more than 200,000 images and 80 object categories (Table 1). The model takes in image as an input and the output contains the bounding boxes, classes and confident scores of the detected objects.

<p style="text-align: center;"><b>Table 1. Class ID and name</b></p>

| Class ID | Class name    | Class ID | Class name     | Class ID | Class name   | Class ID | Class name   |
| -------- | ------------- | -------- | -------------- | -------- | ------------ | -------- | ------------ |
| 1        | person        | 21       | elephant       | 41       | wine glass   | 61       | dining table |
| 2        | bicycle       | 22       | bear           | 42       | cup          | 62       | toilet       |
| 3        | car           | 23       | zebra          | 43       | fork         | 63       | tv           |
| 4        | motorcycle    | 24       | giraffe        | 44       | knife        | 64       | laptop       |
| 5        | airplane      | 25       | backpack       | 45       | spoon        | 65       | mouse        |
| 6        | bus           | 26       | umbrella       | 46       | bowl         | 66       | remote       |
| 7        | train         | 27       | handbag        | 47       | banana       | 67       | keyboard     |
| 8        | truck         | 28       | tie            | 48       | apple        | 68       | cell phone   |
| 9        | boat          | 29       | suitcase       | 49       | sandwich     | 69       | microwave    |
| 10       | traffic light | 30       | frisbee        | 50       | orange       | 70       | oven         |
| 11       | fire hydrant  | 31       | skis           | 51       | broccoli     | 71       | toaster      |
| 12       | stop sign     | 32       | snowboard      | 52       | carrot       | 72       | sink         |
| 13       | parking meter | 33       | sports ball    | 53       | hot dog      | 73       | refrigerator |
| 14       | bench         | 34       | kite           | 54       | pizza        | 74       | book         |
| 15       | bird          | 35       | baseball bat   | 55       | donut        | 75       | clock        |
| 16       | cat           | 36       | baseball glove | 56       | cake         | 76       | vase         |
| 17       | dog           | 37       | skateboard     | 57       | chair        | 77       | scissors     |
| 18       | horse         | 38       | surfboard      | 58       | couch        | 78       | teddy bear   |
| 19       | sheep         | 39       | tennis racket  | 59       | potted plant | 79       | hair drier   |
| 20       | cow           | 40       | bottle         | 60       | bed          | 80       | toothbrush   |
