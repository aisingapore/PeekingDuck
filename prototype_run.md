# Multiprocessing prototype

## Important Note
Things to note, `Runner` is changed, `Pipeline` is not used, `bbox.py` and `ultils.bbox.py` (draw_bbox) returns a frame.

## Running Multiprocessing

Create new environment, install required packages and run multiprocess.
```bash
conda create -n testenv python=3.7
pip install -r requirements.txt
pip install multiprocess
python prototype_run.py
```

## Findings and Conclusions

### Prototype design overview

The Runner is slightly changed to spawn processes to run the nodes in parallel. A wrapper function is written to run the nodes in a while loop within the spawned process. As a spawn process is meant to run a function once, the while loop allows for the node.run to be called more than once. The nodes has to be initiated inside the wrapper function due to some limitation of pickling ABC meta class during multiprocesssing.

The data structure used for multiprocessing has to change, so in the prototype Pipeline was not used. `Multiprocess.Manager().Dict()` was used as the data lake to share the data between processes. Because of this data structure, it is not possible to subset the data lake for individual nodes. 

<u>Problems with data struture</u>

Initially it is planned to use a nested queue to share information between the processes. With the producer node putting frames into the queue and consumers retreive the frame in a FIFO manner. However, the nested queue inside `Multiprocess.Manager().Dict()` actually makes the running peekingduck lag. 

There seems to be some overhead when accessing a nested shared data struture using the data structured offered in multiprocess package. So the nodes that are running in parallel seem to not run as fast as expected.

Also, the data within the dictionary is not thread safe when being updated. So when 1 node is consuming the data and another node is updating the data it causes some weird interactions.

In this prototype, the queues are not used, so the data lake is implemented with `Multiprocess.Manager().Dict()` and the data is shared as a single state (like current implementation in PKD). When the prototype is run, the fps shown on the screen node seems real time, this shows that it is working however it occasionaly will crash at a specific error. 

```bash
Traceback (most recent call last):
  File "E:\Anaconda3\envs\noPKD\lib\site-packages\multiprocess\process.py", line 297, in _bootstrap
    self.run()
  File "E:\Anaconda3\envs\noPKD\lib\site-packages\multiprocess\process.py", line 99, in run
    self._target(*self._args, **self._kwargs)
  File "e:\PeekingDuck_AIAP\PeekingDuck\peekingduck\mp_runner.py", line 69, in node_runner_special
    results = draw_node.run(shared_data)
  File "e:\PeekingDuck_AIAP\PeekingDuck\peekingduck\pipeline\nodes\draw\bbox.py", line 51, in run
    inputs["bbox_labels"], self.show_labels)  # type: ignore
  File "e:\PeekingDuck_AIAP\PeekingDuck\peekingduck\pipeline\nodes\draw\utils\bbox.py", line 47, in draw_bboxes
    colour = PRIMARY_PALETTE[colour_indx[bbox_labels[i]] % TOTAL_COLOURS]
IndexError: index 1 is out of bounds for axis 0 with size 1
```

The hypothesis is that when the draw node is iterating through the multiple bboxes in `colour = PRIMARY_PALETTE[colour_indx[bbox_labels[i]] % TOTAL_COLOURS]`, the data could be mutated. For example the previous frame had 2 bboxes detected but the current frame only have 1 bbox. This causes an error when iterating to the 2nd bbox which no longer exists.

## Possible future developments

- A suspicion is that the GIL prevents true parallelism. Consider looking at multithreading to see if it is able to do image capture and screen in parallel.
- Datastructure used is possibly mutated by other processes, it may be a good idea to look at a proper implementation of a queue/s to manage peekingduck inferrence. This can involve how individual nodes behave.

Example if csv writer wants to consume a frame, and model is still generating the frame. csv writer should get the frame from live. This helps to ensure the fps is smooth and not being bottlenecked by model.
