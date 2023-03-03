.. include:: /include/substitution.rst

.. raw:: html

   <style>
        .wy-table-responsive table td ul li{
            list-style-type: none;
        }

        div.install .training-framework-btn {
            display: inline-block;
            margin: 12px 6px;
            padding: 5px 11px;
            background-color: #fff3cd;
            border: none;
            border-radius: 3px;
            color: black;
            font-size: 90%;
            font-family: "Nunito", sans-serif;
            font-weight: 400;
        }

        div.install .training-framework-btn.active {
            background-color: #a76d60;
            color: white;
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
     <button class="training-framework-btn training-framework-py active">Pytorch</button>
     <button class="training-framework-btn training-framework-tf">Tensorflow</button>
   </div>
   <br><br>

We are using the yaml syntax for the config file

Data
====


.. raw:: html

   <h4>Train Test Split</h4>

   Split arrays or matrices into random train and test subsets.

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
                    <td><p>Quick utility that wraps input validation, next(ShuffleSplit().split(X, y)), and application to input data into a single call for splitting (and optionally subsampling) data into a one-liner.</p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td><code class="xref"><span class="pre">_partial_</span></code></td>
                    <td><p>True</p></td>
                    <td><p>Partial initialization of function to allow stratify in dataset.</p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td><code class="xref"><span class="pre">test_size</span></code></td>
                    <td><p>0.125</p></td>
                    <td><p>If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.</p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td><code class="xref"><span class="pre">shuffle</span></code></td>
                    <td><p>True</p></td>
                    <td><p>Whether or not to shuffle the data before splitting.</p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td><code class="xref"><span class="pre">random_state</span></code></td>
                    <td><p>${random_state}</p></td>
                    <td><p>Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.</p></td>
                </tr>

            </tbody>
        </table>
   </div>

.. raw:: html

   <div class="install">
     <strong>Framework</strong>
     <button class="training-framework-btn training-framework-py active">Pytorch</button>
     <button class="training-framework-btn training-framework-tf">Tensorflow</button>
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

      https://www.cs.toronto.edu/~kriz/cifar.html. The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

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
                        <td><p>csv file column name to image path. Allow absolute path or relative path.</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">target_col_name</span></code></td>
                        <td><p>"class_name"</p></td>
                        <td><p>csv file column name to target string.</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">target_col_id</span></code></td>
                        <td><p>"class_id"</p></td>
                        <td><p>csv file column name to target integer.</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">stratify_by</span></code></td>
                        <td><p>"${.target_col_name}"</p></td>
                        <td><p>csv file column name to be stratify by.</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">classification_type</span></code></td>
                        <td><p>"multiclass"</p></td>
                        <td><p>Task type.</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">image_size</span></code></td>
                        <td><p>224</p></td>
                        <td><p>resized image pixel size.</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">num_classes</span></code></td>
                        <td><p>10</p></td>
                        <td><p>number of classes.</p></td>
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
                        <td><p>dict mapping between target `string` and `integer`.</p></td>
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
                        <td><p>list of classes name</p></td>
                    </tr>
                </tbody>
            </table>
         </div>

   .. tab:: Vegfru

      Based on dataset from https://github.com/ustc-vim/vegfru. For the paper "VegFru: A Domain-Specific Dataset for Fine-grained Visual Categorization".

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
                        <td><p>csv file column name to image path. Allow absolute path or relative path.</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">target_col_name</span></code></td>
                        <td colspan="3"><p>"class_name"</p></td>
                        <td><p>csv file column name to target string.</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">target_col_id</span></code></td>
                        <td colspan="3"><p>"class_id"</p></td>
                        <td><p>csv file column name to target integer.</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">stratify_by</span></code></td>
                        <td><p>class_id</p></td>
                        <td><p>class_id</p></td>
                        <td><p>class_name</p></td>
                        <td><p>csv file column name to be stratify by.</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">classification_type</span></code></td>
                        <td colspan="3"><p>"multiclass"</p></td>
                        <td><p>Task type.</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">image_size</span></code></td>
                        <td colspan="3"><p>224</p></td>
                        <td><p>image pixel size.</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">num_classes</span></code></td>
                        <td><p>5</p></td>
                        <td><p>15</p></td>
                        <td><p>25</p></td>
                        <td><p>number of classes.</p></td>
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
                        <td><p>dict mapping between target `string` and `integer`.</p></td>
                    </tr>
                </tbody>
            </table>
         </div>

   .. tab:: RSNA

      https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data. The goal of this dataset is to identify cases of breast cancer in mammograms from screening exams. It is important to identify cases of cancer for obvious reasons, but false positives also have downsides for patients. As millions of women get mammograms each year, a useful machine learning tool could help a great many people.

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
                        <td><p>csv file column name to image path. Allow absolute path or relative path.</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">target_col_name</span></code></td>
                        <td><p>"class_name" #class_name cancer</p></td>
                        <td><p>csv file column name to target string.</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">target_col_id</span></code></td>
                        <td><p>"class_id"</p></td>
                        <td><p>csv file column name to target integer.</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">stratify_by</span></code></td>
                        <td><p>"${.target_col_name}"</p></td>
                        <td><p>csv file column name to be stratify by.</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">classification_type</span></code></td>
                        <td><p>"multiclass"</p></td>
                        <td><p>Task type.</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">image_size</span></code></td>
                        <td><p>224</p></td>
                        <td><p>resized image pixel size.</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">num_classes</span></code></td>
                        <td><p>2</p></td>
                        <td><p>number of classes.</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">class_name_to_id</span></code></td>
                        <td><ul>
                            <li>benign: 0</li>
                            <li>malignant: 1</li>
                        </ul></td>
                        <td><p>dict mapping between target `string` and `integer`.</p></td>
                    </tr>
                </tbody>
            </table>
         </div>



Model
=====

.. raw:: html

   <div class="install">
     <strong>Framework</strong>
     <button class="training-framework-btn training-framework-py active">Pytorch</button>
     <button class="training-framework-btn training-framework-tf">Tensorflow</button>
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
                        <th colspan="2" class="head"><p>Key</p></th>
                        <th class="head"><p>Value</p></th>
                        <th class="head"><p>Description</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">adapter</span></code></td>
                        <td><p>"timm"</p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">task</span></code></td>
                        <td><p>"classification"</p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">model_name</span></code></td>
                        <td><p>"vgg16"</p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">weights</span></code></td>
                        <td><p>"DEFAULT"</p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">pretrained</span></code></td>
                        <td><p>True</p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">fine_tune</span></code></td>
                        <td><p>True</p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">fine_tune_modules</span></code></td>
                        <td><p></p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">features</span></code></td>
                        <td><p>7</p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">pre_logits</span></code></td>
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
                        <td colspan="2"><code class="xref"><span class="pre">num_classes</span></code></td>
                        <td><p>${data_module.dataset.num_classes}</p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">device</span></code></td>
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
                    <td><code class="xref"><span class="pre">entity</span></code></td>
                    <td>"peekingduck"</td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><code class="xref"><span class="pre">project</span></code></td>
                    <td>"${project_name}"</td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><code class="xref"><span class="pre">run_name</span></code></td>
                    <td>"${stores.unique_id}"</td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><code class="xref"><span class="pre">framework</span></code></td>
                    <td>"${framework}"</td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><code class="xref"><span class="pre">debug</span></code></td>
                    <td>${debug}</td>
                    <td><p></p></td>
                </tr>
            </tbody>
        </table>
   </div>


Trainer
=======

.. raw:: html

   <div class="install">
     <strong>Framework</strong>
     <button class="training-framework-btn training-framework-py active">Pytorch</button>
     <button class="training-framework-btn training-framework-tf">Tensorflow</button>
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
     <strong>Framework</strong>
     <button class="training-framework-btn training-framework-py active">Pytorch</button>
     <button class="training-framework-btn training-framework-tf">Tensorflow</button>
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
                        <th colspan="2" class="head"><p>Key</p></th>
                        <th class="head"><p>Value</p></th>
                        <th class="head"><p>Description</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">Accuracy</span></code></td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">average</span></code></td>
                        <td>"micro"</td>
                        <td></td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">Precision</span></code></td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">average</span></code></td>
                        <td>"macro"</td>
                        <td></td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">Recall</span></code></td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">AUROC</span></code></td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">average</span></code></td>
                        <td>"weighted"</td>
                        <td></td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">CalibrationError</span></code></td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">norm</span></code></td>
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
                        <th colspan="2" class="head"><p>Key</p></th>
                        <th class="head"><p>Value</p></th>
                        <th class="head"><p>Description</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">CategoricalAccuracy</span></code></td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">Precision</span></code></td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">thresholds</span></code></td>
                        <td>0.5</td>
                        <td></td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">Recall</span></code></td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">thresholds</span></code></td>
                        <td>0.5</td>
                        <td></td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">AUC</span></code></td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">from_logits</span></code></td>
                        <td>False</td>
                        <td></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">multi_label</span></code></td>
                        <td>True</td>
                        <td></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">num_labels</span></code></td>
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
     <strong>Framework</strong>
     <button class="training-framework-btn training-framework-py active">Pytorch</button>
     <button class="training-framework-btn training-framework-tf">Tensorflow</button>
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
                        <th colspan="2" class="head"><p>Key</p></th>
                        <th class="head"><p>Value</p></th>
                        <th class="head"><p>Description</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="row-even">
                        <td colspan="2"><code class="xref"><span class="pre">Logger</span></code></td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr class="row-odd">
                        <td colspan="2"><code class="xref"><span class="pre">EarlyStopping</span></code></td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr class="row-even">
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">mode</span></code></td>
                        <td>"max"</td>
                        <td></td>
                    </tr>
                    <tr class="row-odd">
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">monitor</span></code></td>
                        <td>"val_MulticlassAccuracy"</td>
                        <td></td>
                    </tr>
                    <tr class="row-even">
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">patience</span></code></td>
                        <td>${trainer.pytorch.global_train_params.patience}</td>
                        <td></td>
                    </tr>
                    <tr class="row-odd">
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">min_delta</span></code></td>
                        <td>0.000001 # 1e-6</td>
                        <td></td>
                    </tr>
                    <tr class="row-even">
                        <td colspan="2"><code class="xref"><span class="pre">History</span></code></td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr class="row-odd">
                        <td colspan="2"><code class="xref"><span class="pre">MetricMeter</span></code></td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr class="row-even">
                        <td colspan="2"><code class="xref"><span class="pre">ModelCheckpoint</span></code></td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr class="row-odd">
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">mode</span></code></td>
                        <td>"max"</td>
                        <td></td>
                    </tr>
                    <tr class="row-even">
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">monitor</span></code></td>
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
                        <th colspan="2" class="head"><p>Key</p></th>
                        <th class="head"><p>Value</p></th>
                        <th class="head"><p>Description</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="row-even">
                        <td colspan="2"><code class="xref"><span class="pre">EarlyStopping</span></code></td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr class="row-odd">
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">patience</span></code></td>
                        <td>3</td>
                        <td></td>
                    </tr>
                    <tr class="row-even">
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">restore_best_weights</span></code></td>
                        <td>True</td>
                        <td></td>
                    </tr>
                    <tr class="row-odd">
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">monitor</span></code></td>
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
                    <td><code class="xref"><span class="pre">unique_id</span></code></td>
                    <td><p>${now:%Y%m%d_%H%M%S}</p></td>
                    <td></td>
                </tr>
            </tbody>
        </table>
   </div>