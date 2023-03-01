.. include:: /include/substitution.rst

.. raw:: html

   <script>

   </script>

*********************
Image Classification
*********************

.. raw:: html

   <div class="install">
     <strong>Framework</strong>
     <input type="radio" name="framework" id="training-framework-py" checked="checked">
     <label for="training-framework-py">Pytorch</label>
     <input type="radio" name="framework" id="training-framework-tf">
     <label for="training-framework-tf">Tensorflow</label>
   </div>
   <br><br>

We are using the yaml syntax for the config file

Data
====

.. tabs::

   .. tab:: Cifar10

      .. tabs::

         .. tab:: General

            Hello welcome to statquest, this is josh stammer and today...

            .. raw:: html

                <div class="wy-table-responsive">
                    <table class="docutils align-default">
                        <thead>
                            <tr class="row-odd">
                                <th class="head"><p>Cifar10</p></th>
                                <th class="head"><p>Context</p></th>
                                <th class="head"><p>Example</p></th>
                                
                            </tr>
                        </thead>
                        <tbody>
                            <tr class="row-even">
                                <td><p><span class="blue">Blue</span></p></td>
                                <td><p>Current folder</p></td>
                                <td><p><span class="blue">[~user/src]</span></p></td>
                                <td><p><span class="blue">[~user/src]</span></p></td>
                            </tr>
                            <tr class="row-odd"><td><p><span class="green">Green</span></p></td>
                                <td><p>User input: what you type in</p></td>
                                <td><p>&gt; <span class="green">peekingduck --version</span></p></td>
                                <td><p>&gt; <span class="green">peekingduck --version</span></p></td>
                            </tr>
                            <tr class="row-even"><td><p>Black</p></td>
                                <td><p>PeekingDuck’s output</p></td>
                                <td><p>peekingduck, version v1.2.0</p></td>
                                <td><p>peekingduck, version v1.2.0</p></td>
                            </tr>
                        </tbody>
                    </table>
                </div>

         .. tab:: Dataset

            The closest star to us.

         .. tab:: Train Test Split

            The second closest star to us.

         .. tab:: Transform

            The North Star.

   .. tab:: Vegfru

      .. tabs::

         .. tab:: General

            .. raw:: html

                <div class="wy-table-responsive">
                    <table class="docutils align-default">
                        <thead>
                            <tr class="row-odd">
                                <th class="head"><p>Vegfru</p></th>
                                <th class="head"><p>Context</p></th>
                                <th class="head"><p>Example</p></th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr class="row-even">
                                <td><p><span class="blue">Blue</span></p></td>
                                <td><p>Current folder</p></td>
                                <td><p><span class="blue">[~user/src]</span></p></td>
                            </tr>
                            <tr class="row-odd"><td><p><span class="green">Green</span></p></td>
                                <td><p>User input: what you type in</p></td>
                                <td><p>&gt; <span class="green">peekingduck --version</span></p></td>
                            </tr>
                            <tr class="row-even"><td><p>Black</p></td>
                                <td><p>PeekingDuck’s output</p></td>
                                <td><p>peekingduck, version v1.2.0</p></td>
                            </tr>
                        </tbody>
                    </table>
                </div>

         .. tab:: Dataset

            The closest star to us.

         .. tab:: Train Test Split

            The second closest star to us.

         .. tab:: Transform

            The North Star.

   .. tab:: RSNA

      .. tabs::

         .. tab:: General

            .. raw:: html

                <div class="wy-table-responsive">
                    <table class="docutils align-default">
                        <thead>
                            <tr class="row-odd">
                                <th class="head"><p>RSNA</p></th>
                                <th class="head"><p>Context</p></th>
                                <th class="head"><p>Example</p></th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr class="row-even">
                                <td><p><span class="blue">Blue</span></p></td>
                                <td><p>Current folder</p></td>
                                <td><p><span class="blue">[~user/src]</span></p></td>
                            </tr>
                            <tr class="row-odd"><td><p><span class="green">Green</span></p></td>
                                <td><p>User input: what you type in</p></td>
                                <td><p>&gt; <span class="green">peekingduck --version</span></p></td>
                            </tr>
                            <tr class="row-even"><td><p>Black</p></td>
                                <td><p>PeekingDuck’s output</p></td>
                                <td><p>peekingduck, version v1.2.0</p></td>
                            </tr>
                        </tbody>
                    </table>
                </div>

         .. tab:: Dataset

            The closest star to us.

         .. tab:: Train Test Split

            The second closest star to us.

         .. tab:: Transform

            The North Star.


Model
=====

.. raw:: html

   <div class="install">
     <div class="pkd-expandable training-pytorch">
        <h4>Pytorch</h4>
        <div class="wy-table-responsive">
            <table class="docutils align-default">
                <thead>
                    <tr class="row-odd">
                        <th class="head"><p>Pytorch</p></th>
                        <th class="head"><p>Context</p></th>
                        <th class="head"><p>Example</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="row-even">
                        <td><p><span class="blue">Blue</span></p></td>
                        <td><p>Current folder</p></td>
                        <td><p><span class="blue">[~user/src]</span></p></td>
                    </tr>
                    <tr class="row-odd"><td><p><span class="green">Green</span></p></td>
                        <td><p>User input: what you type in</p></td>
                        <td><p>&gt; <span class="green">peekingduck --version</span></p></td>
                    </tr>
                    <tr class="row-even"><td><p>Black</p></td>
                        <td><p>PeekingDuck’s output</p></td>
                        <td><p>peekingduck, version v1.2.0</p></td>
                    </tr>
                </tbody>
            </table>
        </div>
     
     </div>
     <div class="pkd-expandable training-tensorflow">
        <h4>Tensorflow</h4>
        <div class="wy-table-responsive">
            <table class="docutils align-default">
                <thead>
                    <tr class="row-odd">
                        <th class="head"><p>Tensorflow</p></th>
                        <th class="head"><p>Context</p></th>
                        <th class="head"><p>Example</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="row-even">
                        <td><p><span class="blue">Blue</span></p></td>
                        <td><p>Current folder</p></td>
                        <td><p><span class="blue">[~user/src]</span></p></td>
                    </tr>
                    <tr class="row-odd"><td><p><span class="green">Green</span></p></td>
                        <td><p>User input: what you type in</p></td>
                        <td><p>&gt; <span class="green">peekingduck --version</span></p></td>
                    </tr>
                    <tr class="row-even"><td><p>Black</p></td>
                        <td><p>PeekingDuck’s output</p></td>
                        <td><p>peekingduck, version v1.2.0</p></td>
                    </tr>
                </tbody>
            </table>
        </div>

     </div>
   </div>


Model Analysis
==============

.. raw:: html

   <div class="install">
     <div class="pkd-expandable training-pytorch">
        <h4>Pytorch</h4>
        <div class="wy-table-responsive">
            <table class="docutils align-default">
                <thead>
                    <tr class="row-odd">
                        <th class="head"><p>Pytorch</p></th>
                        <th class="head"><p>Context</p></th>
                        <th class="head"><p>Example</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="row-even">
                        <td><p><span class="blue">Blue</span></p></td>
                        <td><p>Current folder</p></td>
                        <td><p><span class="blue">[~user/src]</span></p></td>
                    </tr>
                    <tr class="row-odd"><td><p><span class="green">Green</span></p></td>
                        <td><p>User input: what you type in</p></td>
                        <td><p>&gt; <span class="green">peekingduck --version</span></p></td>
                    </tr>
                    <tr class="row-even"><td><p>Black</p></td>
                        <td><p>PeekingDuck’s output</p></td>
                        <td><p>peekingduck, version v1.2.0</p></td>
                    </tr>
                </tbody>
            </table>
        </div>
     
     </div>
     <div class="pkd-expandable training-tensorflow">
        <h4>Tensorflow</h4>
        <div class="wy-table-responsive">
            <table class="docutils align-default">
                <thead>
                    <tr class="row-odd">
                        <th class="head"><p>Tensorflow</p></th>
                        <th class="head"><p>Context</p></th>
                        <th class="head"><p>Example</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="row-even">
                        <td><p><span class="blue">Blue</span></p></td>
                        <td><p>Current folder</p></td>
                        <td><p><span class="blue">[~user/src]</span></p></td>
                    </tr>
                    <tr class="row-odd"><td><p><span class="green">Green</span></p></td>
                        <td><p>User input: what you type in</p></td>
                        <td><p>&gt; <span class="green">peekingduck --version</span></p></td>
                    </tr>
                    <tr class="row-even"><td><p>Black</p></td>
                        <td><p>PeekingDuck’s output</p></td>
                        <td><p>peekingduck, version v1.2.0</p></td>
                    </tr>
                </tbody>
            </table>
        </div>

     </div>
   </div>


Trainer
=======

.. raw:: html

   <div class="install">
     <div class="pkd-expandable training-pytorch">
        <h4>Pytorch</h4>
        <div class="wy-table-responsive">
            <table class="docutils align-default">
                <thead>
                    <tr class="row-odd">
                        <th class="head"><p>Pytorch</p></th>
                        <th class="head"><p>Context</p></th>
                        <th class="head"><p>Example</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="row-even">
                        <td><p><span class="blue">Blue</span></p></td>
                        <td><p>Current folder</p></td>
                        <td><p><span class="blue">[~user/src]</span></p></td>
                    </tr>
                    <tr class="row-odd"><td><p><span class="green">Green</span></p></td>
                        <td><p>User input: what you type in</p></td>
                        <td><p>&gt; <span class="green">peekingduck --version</span></p></td>
                    </tr>
                    <tr class="row-even"><td><p>Black</p></td>
                        <td><p>PeekingDuck’s output</p></td>
                        <td><p>peekingduck, version v1.2.0</p></td>
                    </tr>
                </tbody>
            </table>
        </div>
     
     </div>
     <div class="pkd-expandable training-tensorflow">
        <h4>Tensorflow</h4>
        <div class="wy-table-responsive">
            <table class="docutils align-default">
                <thead>
                    <tr class="row-odd">
                        <th class="head"><p>Tensorflow</p></th>
                        <th class="head"><p>Context</p></th>
                        <th class="head"><p>Example</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="row-even">
                        <td><p><span class="blue">Blue</span></p></td>
                        <td><p>Current folder</p></td>
                        <td><p><span class="blue">[~user/src]</span></p></td>
                    </tr>
                    <tr class="row-odd"><td><p><span class="green">Green</span></p></td>
                        <td><p>User input: what you type in</p></td>
                        <td><p>&gt; <span class="green">peekingduck --version</span></p></td>
                    </tr>
                    <tr class="row-even"><td><p>Black</p></td>
                        <td><p>PeekingDuck’s output</p></td>
                        <td><p>peekingduck, version v1.2.0</p></td>
                    </tr>
                </tbody>
            </table>
        </div>

     </div>
   </div>

Metrics
=======

.. raw:: html

   <div class="install">
     <div class="pkd-expandable training-pytorch">
        <h4>Pytorch</h4>
        <div class="wy-table-responsive">
            <table class="docutils align-default">
                <thead>
                    <tr class="row-odd">
                        <th class="head"><p>Key</p></th>
                        <th class="head"><p>Value</p></th>
                        <th class="head"><p>Description</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Accuracy</td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr>
                        <td>Accuracy.average</td>
                        <td>"micro"</td>
                        <td></td>
                    </tr>
                    <tr>
                        <td>Precision</td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr>
                        <td>Precision.average</td>
                        <td>"macro"</td>
                        <td></td>
                    </tr>
                    <tr>
                        <td>Recall</td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr>
                        <td>AUROC</td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr>
                        <td>AUROC.average</td>
                        <td>"weighted"</td>
                        <td></td>
                    </tr>
                    <tr>
                        <td>CalibrationError</td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr>
                        <td>CalibrationError.norm</td>
                        <td>"l1"</td>
                        <td></td>
                    </tr>
                </tbody>
            </table>
        </div>
     
     </div>
     <div class="pkd-expandable training-tensorflow">
        <h4>Tensorflow</h4>
        <div class="wy-table-responsive">
            <table class="docutils align-default">
                <thead>
                    <tr class="row-odd">
                        <th class="head"><p>Key</p></th>
                        <th class="head"><p>Value</p></th>
                        <th class="head"><p>Description</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>CategoricalAccuracy</td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr>
                        <td>Precision</td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr>
                        <td>Precision.thresholds</td>
                        <td>0.5</td>
                        <td></td>
                    </tr>
                    <tr>
                        <td>Recall</td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr>
                        <td>Recall.thresholds</td>
                        <td>0.5</td>
                        <td></td>
                    </tr>
                    <tr>
                        <td>AUC</td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr>
                        <td>AUC.from_logits</td>
                        <td>False</td>
                        <td></td>
                    </tr>
                    <tr>
                        <td>AUC.multi_label</td>
                        <td>True</td>
                        <td></td>
                    </tr>
                    <tr>
                        <td>AUC.num_labels</td>
                        <td>${data_module.dataset.num_classes}</td>
                        <td></td>
                    </tr>
                </tbody>
            </table>
        </div>

     </div>
   </div>


Callbacks
=========

.. raw:: html

   <div class="install">
     <div class="pkd-expandable training-pytorch">
        <h4>Pytorch</h4>
        <p>The table below shows the default values:</p>
        <p>These are the onlky available callbacks as of v1.0: EarlyStopping | History | MetricMeter | ModelCheckpoint</p>
        <div class="wy-table-responsive">
            <table class="docutils align-default">
                <thead>
                    <tr class="row-odd">
                        <th class="head"><p>Key</p></th>
                        <th class="head"><p>Value</p></th>
                        <th class="head"><p>Description</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="row-even">
                        <td>Logger</td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr class="row-odd">
                        <td>EarlyStopping</td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr class="row-even">
                        <td>EarlyStopping.mode</td>
                        <td>"max"</td>
                        <td></td>
                    </tr>
                    <tr class="row-odd">
                        <td>EarlyStopping.monitor</td>
                        <td>"val_MulticlassAccuracy"</td>
                        <td></td>
                    </tr>
                    <tr class="row-even">
                        <td>EarlyStopping.patience</td>
                        <td>${trainer.pytorch.global_train_params.patience}</td>
                        <td></td>
                    </tr>
                    <tr class="row-odd">
                        <td>EarlyStopping.min_delta</td>
                        <td>0.000001 # 1e-6</td>
                        <td></td>
                    </tr>
                    <tr class="row-even">
                        <td>History</td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr class="row-odd">
                        <td>MetricMeter</td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr class="row-even">
                        <td>ModelCheckpoint</td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr class="row-odd">
                        <td>ModelCheckpoint.mode</td>
                        <td>"max"</td>
                        <td></td>
                    </tr>
                    <tr class="row-even">
                        <td>ModelCheckpoint.monitor</td>
                        <td>"val_MulticlassAccuracy"</td>
                        <td></td>
                    </tr>
                </tbody>
            </table>
        </div>
     
     </div>
     <div class="pkd-expandable training-tensorflow">
        <h4>Tensorflow</h4>
        <p>The table below shows the default values:</p>
        <p>Check out the <a href="https://www.tensorflow.org/api_docs/python/tf/keras/callbacks">TensorFlow v2 Callbacks</a> documentation for more details</p>
        <div class="wy-table-responsive">
            <table class="docutils align-default">
                <thead>
                    <tr class="row-odd">
                        <th class="head"><p>Key</p></th>
                        <th class="head"><p>Value</p></th>
                        <th class="head"><p>Description</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="row-even">
                        <td>EarlyStopping</td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr class="row-odd">
                        <td>EarlyStopping.patience</td>
                        <td>3</td>
                        <td></td>
                    </tr>
                    <tr class="row-even">
                        <td>EarlyStopping.restore_best_weights</td>
                        <td>True</td>
                        <td></td>
                    </tr>
                    <tr class="row-odd">
                        <td>EarlyStopping.monitor</td>
                        <td>"val_categorical_accuracy"</td>
                        <td></td>
                    </tr>
                </tbody>
            </table>
        </div>

     </div>
   </div>


Store
=====

.. raw:: html

   <div class="install">
     <div class="pkd-expandable training-pytorch">
        <h4>Pytorch</h4>
        <div class="wy-table-responsive">
            <table class="docutils align-default">
                <thead>
                    <tr class="row-odd">
                        <th class="head"><p>Pytorch</p></th>
                        <th class="head"><p>Context</p></th>
                        <th class="head"><p>Example</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="row-even">
                        <td><p><span class="blue">Blue</span></p></td>
                        <td><p>Current folder</p></td>
                        <td><p><span class="blue">[~user/src]</span></p></td>
                    </tr>
                    <tr class="row-odd"><td><p><span class="green">Green</span></p></td>
                        <td><p>User input: what you type in</p></td>
                        <td><p>&gt; <span class="green">peekingduck --version</span></p></td>
                    </tr>
                    <tr class="row-even"><td><p>Black</p></td>
                        <td><p>PeekingDuck’s output</p></td>
                        <td><p>peekingduck, version v1.2.0</p></td>
                    </tr>
                </tbody>
            </table>
        </div>
     
     </div>
     <div class="pkd-expandable training-tensorflow">
        <h4>Tensorflow</h4>
        <div class="wy-table-responsive">
            <table class="docutils align-default">
                <thead>
                    <tr class="row-odd">
                        <th class="head"><p>Tensorflow</p></th>
                        <th class="head"><p>Context</p></th>
                        <th class="head"><p>Example</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="row-even">
                        <td><p><span class="blue">Blue</span></p></td>
                        <td><p>Current folder</p></td>
                        <td><p><span class="blue">[~user/src]</span></p></td>
                    </tr>
                    <tr class="row-odd"><td><p><span class="green">Green</span></p></td>
                        <td><p>User input: what you type in</p></td>
                        <td><p>&gt; <span class="green">peekingduck --version</span></p></td>
                    </tr>
                    <tr class="row-even"><td><p>Black</p></td>
                        <td><p>PeekingDuck’s output</p></td>
                        <td><p>peekingduck, version v1.2.0</p></td>
                    </tr>
                </tbody>
            </table>
        </div>

     </div>
   </div>