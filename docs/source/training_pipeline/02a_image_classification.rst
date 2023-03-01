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

Data
====

.. tabs::

   .. tab:: Cifar10

      .. tabs::

         .. tab:: General

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


Callbacks
=========

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