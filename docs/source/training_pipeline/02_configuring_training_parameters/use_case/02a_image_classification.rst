.. include:: /include/substitution.rst

.. raw:: html

   <style>
        .wy-table-responsive table td ul li{
            list-style-type: none;
        }
   </style>

*********************
Image Classification
*********************

.. toctree::
   :maxdepth: 3

   /training_pipeline/02_configuring_training_parameters/use_case/02a_image_classification

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


.. raw:: html

   <h4>Train Test Split</h4>

   This is a statement about Train Test Split

   <div class="wy-table-responsive">
        <table class="docutils align-default">
            <thead>
                <tr class="row-odd">
                    <th colspan="2" class="head"><p>Key</p></th>
                    <th class="head"><p>Value</p></th>
                    <th class="head"><p>Description</p></th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td colspan="2"><code class="xref"><span class="pre">resample_strategy</span></code></td>
                    <td><p></p></td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td><code class="xref"><span class="pre">_target_</span></code></td>
                    <td><p>sklearn.model_selection.train_test_split</p></td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td><code class="xref"><span class="pre">_partial_</span></code></td>
                    <td><p>True</p></td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td><code class="xref"><span class="pre">test_size</span></code></td>
                    <td><p>0.125</p></td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td><code class="xref"><span class="pre">shuffle</span></code></td>
                    <td><p>True</p></td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td><code class="xref"><span class="pre">random_state</span></code></td>
                    <td><p>${random_state}</p></td>
                    <td><p></p></td>
                </tr>

            </tbody>
        </table>
   </div>

.. raw:: html

   <div class="install">
     <input type="radio" name="framework" id="training-framework3" class="training-framework-py" checked="checked">
     <label for="training-framework-py">Pytorch</label>
     <input type="radio" name="framework" id="training-framework4" class="training-framework-tf">
     <label for="training-framework-tf">Tensorflow</label>
   </div>
   <br><br>

.. raw:: html

   <div class="install">
      <div class="pkd-expandable training-pytorch">
        <h4>Pytorch</h4>

.. include:: /training_pipeline/02_configuring_training_parameters/data_module/pytorch_config.rst

.. raw:: html

   </div>
      <div class="pkd-expandable training-tensorflow">
        <h4>Tensorflow</h4>

.. include:: /training_pipeline/02_configuring_training_parameters/data_module/tensorflow_config.rst

.. raw:: html

   </div></div>

.. raw:: html

   <h4>Default Datasets</h4>

.. tabs::

   .. tab:: Cifar10

      This is a statement about Cifar10

      .. raw:: html

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
                        <td><code class="xref"><span class="pre">download</span></code></td>
                        <td><p>False</p></td>
                        <td><p>Download from url below</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">url</span></code></td>
                        <td><p>"https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"</p></td>
                        <td><p>URL to download dataset</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">blob_file</span></code></td>
                        <td><p>"cifar10.zip"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">root_dir</span></code></td>
                        <td><p>"data"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">train_dir</span></code></td>
                        <td><p>"./${.root_dir}/${project_name}"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">test_dir</span></code></td>
                        <td><p>"./${.root_dir}/${project_name}"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">train_csv</span></code></td>
                        <td><p>"./${.root_dir}/${project_name}/train.csv"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">image_path_col_name</span></code></td>
                        <td><p>"image_path"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">target_col_name</span></code></td>
                        <td><p>"class_name" #class_id</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">target_col_id</span></code></td>
                        <td><p>"class_id"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">stratify_by</span></code></td>
                        <td><p>"${.target_col_name}"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">classification_type</span></code></td>
                        <td><p>"multiclass"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">image_size</span></code></td>
                        <td><p>224</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">num_classes</span></code></td>
                        <td><p>10</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">class_name_to_id</span></code></td>
                        <td><ul>
                                <li>airplane: 0</li>
                                <li>automobile: 1</li>
                                <li>bird: 2</li>
                                <li>cat: 3</li>
                                <li>deer: 4</li>
                                <li>dog: 5</li>
                                <li>frog: 6</li>
                                <li>horse: 7</li>
                                <li>ship: 8</li>
                                <li>truck: 9</li>
                        </ul></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">classes</span></code></td>
                        <td><ul>
                                <li>- airplane</li>
                                <li>- automobile</li>
                                <li>- bird</li>
                                <li>- cat</li>
                                <li>- deer</li>
                                <li>- dog</li>
                                <li>- frog</li>
                                <li>- horse</li>
                                <li>- ship</li>
                                <li>- truck</li>
                        </ul></td>
                        <td><p></p></td>
                    </tr>
                </tbody>
            </table>
         </div>

   .. tab:: Vegfru

      This is a statement about Vegfru

      .. raw:: html

         <div class="wy-table-responsive">
            <table class="docutils align-default">
                <thead>
                    <tr class="row-odd">
                        <th class="head"><p>Key</p></th>
                        <th class="head"><p>VegFru5</p></th>
                        <th class="head"><p>VegFru15</p></th>
                        <th class="head"><p>VegFru25</p></th>
                        <th class="head"><p>Description</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><code class="xref"><span class="pre">download</span></code></td>
                        <td colspan="3"><p>False</p></td>
                        <td><p>Download from url below</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">url</span></code></td>
                        <td colspan="3"><p>""</p></td>
                        <td><p>URL to download dataset</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">blob_file</span></code></td>
                        <td colspan="3"><p>"vegfru.zip"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">root_dir</span></code></td>
                        <td colspan="3"><p>"data"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">train_dir</span></code></td>
                        <td colspan="3"><p>"./${.root_dir}/vegfru"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">test_dir</span></code></td>
                        <td colspan="3"><p>"./${.root_dir}/vegfru"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">train_csv</span></code></td>
                        <td><p>"./${.root_dir}/vegfru/vegfru5.csv"</p></td>
                        <td><p>"./${.root_dir}/vegfru/vegfru15.csv"</p></td>
                        <td><p>"./${.root_dir}/vegfru/train.csv"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">image_path_col_name</span></code></td>
                        <td colspan="3"><p>"image_path"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">target_col_name</span></code></td>
                        <td colspan="3"><p>"class_name"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">target_col_id</span></code></td>
                        <td colspan="3"><p>"class_id"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">stratify_by</span></code></td>
                        <td><p>class_id</p></td>
                        <td><p>class_id</p></td>
                        <td><p>class_name</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">classification_type</span></code></td>
                        <td colspan="3"><p>"multiclass"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">image_size</span></code></td>
                        <td colspan="3"><p>224</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">num_classes</span></code></td>
                        <td><p>5</p></td>
                        <td><p>15</p></td>
                        <td><p>25</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">class_name_to_id</span></code></td>
                        <td><ul>
                                <li>"garlic": 0</li>
                                <li>"cattail": 1</li>
                                <li>"soybean": 2</li>
                                <li>"red_cabbage": 3</li>
                                <li>"mung_bean_sprouts": 4</li>
                        </ul></td>
                        <td><ul>
                                <li>"garlic": 0</li>
                                <li>"cattail": 1</li>
                                <li>"soybean": 2</li>
                                <li>"red_cabbage": 3</li>
                                <li>"mung_bean_sprouts": 4</li>
                                <li>"pakchoi": 5</li>
                                <li>"eggplant": 6</li>
                                <li>"chrysanthemum": 7</li>
                                <li>"snake_gourd": 8</li>
                                <li>"nameko": 9</li>
                                <li>"mustard": 10</li>
                                <li>"Lily": 11</li>
                                <li>"beetroot": 12</li>
                                <li>"kudzu": 13</li>
                                <li>"fallopia_multiflora": 14</li>
                        </ul></td>
                        <td><ul>
                                <li>"alliaceous": 0</li>
                                <li>"aquatic_vegetable": 1</li>
                                <li>"beans": 2</li>
                                <li>"brassia_olreacea": 3</li>
                                <li>"bug_seedling": 4</li>
                                <li>"cabbage": 5</li>
                                <li>"eggplant": 6</li>
                                <li>"green_leafy_vegatable": 7</li>
                                <li>"melon": 8</li>
                                <li>"mushroom": 9</li>
                                <li>"mustard": 10</li>
                                <li>"perennial": 11</li>
                                <li>"root_vegetable": 12</li>
                                <li>"tuber_vagetable": 13</li>
                                <li>"wild_vegetable": 14</li>
                                <li>"berry_fruit": 15</li>
                                <li>"citrus_fruit": 16</li>
                                <li>"collective_fruit": 17</li>
                                <li>"cucurbites": 18</li>
                                <li>"drupe": 19</li>
                                <li>"litchies": 20</li>
                                <li>"nut_fruit": 21</li>
                                <li>"persimmons_and_jujubes_fruit": 22</li>
                                <li>"pome": 23</li>
                                <li>"other_fruit": 24</li>
                        </ul></td>
                        <td><p></p></td>
                    </tr>
                </tbody>
            </table>
         </div>

   .. tab:: RSNA

      This is a statement about RSNA

      .. raw:: html

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
                        <td><code class="xref"><span class="pre">download</span></code></td>
                        <td><p>False</p></td>
                        <td><p>Download from url below</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">url</span></code></td>
                        <td><p></p></td>
                        <td><p>URL to download dataset</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">blob_file</span></code></td>
                        <td><p>"rsna.zip"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">root_dir</span></code></td>
                        <td><p>"data"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">train_dir</span></code></td>
                        <td><p>"./${.root_dir}/${project_name}"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">test_dir</span></code></td>
                        <td><p>"./${.root_dir}/${project_name}"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">train_csv</span></code></td>
                        <td><p>"./${.root_dir}/${project_name}/train.csv"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">image_path_col_name</span></code></td>
                        <td><p>"image_path"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">target_col_name</span></code></td>
                        <td><p>"class_name" #class_name cancer</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">target_col_id</span></code></td>
                        <td><p>"class_id"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">stratify_by</span></code></td>
                        <td><p>"${.target_col_name}"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">classification_type</span></code></td>
                        <td><p>"multiclass"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">image_size</span></code></td>
                        <td><p>224</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">num_classes</span></code></td>
                        <td><p>2</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">class_name_to_id</span></code></td>
                        <td><ul>
                            <li>benign: 0</li>
                            <li>malignant: 1</li>
                        </ul></td>
                        <td><p></p></td>
                    </tr>
                </tbody>
            </table>
         </div>



Model
=====

.. raw:: html

   <div class="install">
     <input type="radio" name="framework" id="training-framework5" class="training-framework-py" checked="checked">
     <label for="training-framework-py">Pytorch</label>
     <input type="radio" name="framework" id="training-framework6" class="training-framework-tf">
     <label for="training-framework-tf">Tensorflow</label>
   </div>
   <br><br>

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
                        <td><code class="xref"><span class="pre">adapter</span></code></td>
                        <td><p>"timm"</p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">task</span></code></td>
                        <td><p>"classification"</p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">model_name</span></code></td>
                        <td><p>"vgg16"</p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">weights</span></code></td>
                        <td><p>"DEFAULT"</p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">pretrained</span></code></td>
                        <td><p>True</p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">fine_tune</span></code></td>
                        <td><p>True</p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">fine_tune_modules</span></code></td>
                        <td><p></p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">fine_tune_modules</span></code>.features</td>
                        <td><p>7</p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">fine_tune_modules</span></code>.pre_logits</td>
                        <td><p>[
                                "fc1",
                                "act1",
                                "drop",
                                "fc2",
                                "act2",
                            ]</p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">num_classes</span></code></td>
                        <td><p>${data_module.dataset.num_classes}</p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">device</span></code></td>
                        <td><p>${device}</p</td>
                        <td><p></p></td>
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
                        <td><code class="xref"><span class="pre">task</span></code></td>
                        <td><p>"classification"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">model_name</span></code></td>
                        <td><p>"VGG16"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">num_classes</span></code></td>
                        <td><p>${data_module.dataset.num_classes}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">image_size</span></code></td>
                        <td><p>${data_module.dataset.image_size}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">device</span></code></td>
                        <td><p>${device}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">pretrained</span></code></td>
                        <td><p>True</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">fine_tune</span></code></td>
                        <td><p>True</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">fine_tune_layers</span></code></td>
                        <td><p>[
                            "prediction_modified",
                            "fc2",
                            "fc1",
                            "block5_conv3",
                            "block5_conv2",
                            "block5_conv1",
                        ]</p></td>
                        <td><p></p></td>
                    </tr>
                </tbody>
            </table>
        </div>

     </div>
   </div>


Model Analysis
==============

.. raw:: html

   <h4>Default Values</h4>
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
                    <td>entity</td>
                    <td>"peekingduck"</td>
                    <td></td>
                </tr>
                <tr>
                    <td>project</td>
                    <td>"${project_name}"</td>
                    <td></td>
                </tr>
                <tr>
                    <td>run_name</td>
                    <td>"${stores.unique_id}"</td>
                    <td></td>
                </tr>
                <tr>
                    <td>framework</td>
                    <td>"${framework}"</td>
                    <td></td>
                </tr>
                <tr>
                    <td>debug</td>
                    <td>${debug}</td>
                    <td></td>
                </tr>
            </tbody>
        </table>
   </div>


Trainer
=======

.. raw:: html

   <div class="install">
     <input type="radio" name="framework" id="training-framework7" class="training-framework-py" checked="checked">
     <label for="training-framework-py">Pytorch</label>
     <input type="radio" name="framework" id="training-framework8" class="training-framework-tf">
     <label for="training-framework-tf">Tensorflow</label>
   </div>
   <br><br>

.. raw:: html

   <div class="install">
      <div class="pkd-expandable training-pytorch">
        <h4>Pytorch</h4>

.. include:: /training_pipeline/02_configuring_training_parameters/trainer/pytorch_config.rst

.. raw:: html

   </div>
      <div class="pkd-expandable training-tensorflow">
        <h4>Tensorflow</h4>

.. include:: /training_pipeline/02_configuring_training_parameters/trainer/tensorflow_config.rst

.. raw:: html

   </div></div>



Metrics
=======

.. raw:: html

   <div class="install">
     <input type="radio" name="framework" id="training-framework9" class="training-framework-py" checked="checked">
     <label for="training-framework-py">Pytorch</label>
     <input type="radio" name="framework" id="training-framework10" class="training-framework-tf">
     <label for="training-framework-tf">Tensorflow</label>
   </div>
   <br><br>

.. raw:: html

   <div class="install">
     <div class="pkd-expandable training-pytorch">
        <h4>Pytorch</h4>
        # Options: Accuracy | Precision | Recall | AUROC | CalibrationError
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
        <p># Options: Accuracy | Precision | Recall | AUC</p>
        <p>Check out the <a href="https://www.tensorflow.org/api_docs/python/tf/keras/metrics">TensorFlow v2 Metrics</a> documentation for more details</p>
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
     <input type="radio" name="framework" id="training-framework11" class="training-framework-py" checked="checked">
     <label for="training-framework-py">Pytorch</label>
     <input type="radio" name="framework" id="training-framework12" class="training-framework-tf">
     <label for="training-framework-tf">Tensorflow</label>
   </div>
   <br><br>

.. raw:: html

   <div class="install">
     <div class="pkd-expandable training-pytorch">
        <h4>Pytorch</h4>
        <p>The table below shows the default values:</p>
        <p>These are the only available callbacks as of v1.0:<br>EarlyStopping | History | MetricMeter | ModelCheckpoint</p>
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

   <h4>Default Values</h4>
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
                    <td><p><span class="">unique_id</span></p></td>
                    <td><p>${now:%Y%m%d_%H%M%S}</p></td>
                    <td></td>
                </tr>
            </tbody>
        </table>
   </div>