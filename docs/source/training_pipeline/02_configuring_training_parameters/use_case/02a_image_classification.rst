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
            background-color: #fff; /* #fff3cd */
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
        .wy-table-responsive table td:last-child, .wy-table-responsive table th:last-child {
            white-space: normal;
            min-width: 450px;
            max-width: 450px;
        }
    </style>

*********************
Image Classification
*********************

* :ref:`config-files-overview`
* :ref:`config-files-mainconfig`
* :ref:`config-files-datamodule`
* :ref:`config-files-model`
* :ref:`config-files-modelanalysis`
* :ref:`config-files-trainer`
* :ref:`config-files-metrics`
* :ref:`config-files-callbacks`
* :ref:`config-files-store`


.. _config-files-overview:

Overview
========

We are using the yaml syntax for the config file. Below is the folder structure and description of how users can understand and navigate the config structure. 
Configuration files that are not user-customizable are not included in the table below.

.. parsed-literal::

    \ :blue:`peekingduck/training/configs/` \ |Blank|
          ├── \ :blue:`callbacks/` \ |Blank|
          │      └── classification.yaml
          ├── \ :blue:`data_module/` \ |Blank|
          │      ├── \ :blue:`data_adapter/` \ |Blank|
          │      │      └── adapter.yaml
          │      ├── \ :blue:`dataset/` \ |Blank|
          │      │      ├── cifar10.yaml
          │      │      ├── rsna.yaml
          │      │      ├── vegfru5.yaml
          │      │      ├── vegfru15.yaml
          │      │      └── vegfru25.yaml
          │      ├── \ :blue:`resample/` \ |Blank|
          │      │      └── train_test_split.yaml
          │      ├── \ :blue:`transform/` \ |Blank|
          │      │      ├── test.yaml
          │      │      └── train.yaml
          │      ├── cifar10.yaml
          │      ├── rsna.yaml
          │      ├── vegfru5.yaml
          │      ├── vegfru15.yaml
          │      └── vegfru25.yaml
          ├── \ :blue:`hydra/` \ |Blank|
          │      └── \ :blue:`job_logging/` \ |Blank|
          │             └── custom.yaml
          ├── \ :blue:`metrics/` \ |Blank|
          │      └── classification.yaml
          ├── \ :blue:`model/` \ |Blank|
          │      └── classification.yaml
          ├── \ :blue:`model_analysis/` \ |Blank|
          │      └── classification.yaml
          ├── \ :blue:`stores/` \ |Blank|
          │      └── classification.yaml
          ├── \ :blue:`trainer/` \ |Blank|
          │      └── classification.yaml
          └── config.yaml


+------------------------------------+---------------------------------------------------------------------------------------------------+
| Folder/file under `configs` folder | Description                                                                                       |
+====================================+===================================================================================================+
| config.yaml                        | Main configuration file for high-level training settings such as                                  |
|                                    |                                                                                                   |
|                                    | project name, framework, debug mode, view mode, etc.                                              |
+------------------------------------+---------------------------------------------------------------------------------------------------+
| data_module                        | Main configuration file for data module.                                                          |
|  └── <name_of_the_dataset>.yaml    |                                                                                                   |
|                                    | The only customizable config is `num_debug_samples`.                                              |
+------------------------------------+---------------------------------------------------------------------------------------------------+
| data_module                        | Controls the batch size for train / test dataset, shuffling control, etc.                         |
|  └── data_adapter                  |                                                                                                   |
+------------------------------------+---------------------------------------------------------------------------------------------------+
| data_module                        | Contains configuration for each dataset in separate .yaml files.                                  |
|  └── dataset                       |                                                                                                   |
|                                    | Customization for the image size and number of classes for image classification.                  |
|                                    |                                                                                                   |
|                                    | User can choose to download the sample dataset when running the pipeline.                         |
|                                    |                                                                                                   |
|                                    | User need to create a separate .yaml file within the folder for custom dataset.                   |
+------------------------------------+---------------------------------------------------------------------------------------------------+
| data_module                        | Controls train/test split and shuffling.                                                          |
|  └── resample                      |                                                                                                   |
+------------------------------------+---------------------------------------------------------------------------------------------------+
| data_module                        | Controls image augmentations / transformations for train / test, such as cropping, flipping, etc. |
|  └── transform                     |                                                                                                   |
+------------------------------------+---------------------------------------------------------------------------------------------------+
| model                              | Controls selection of pre-trained models and fine-tuning model settings                           |
+------------------------------------+---------------------------------------------------------------------------------------------------+
| trainer                            | Control training related parameters including number of epochs,                                   |
|                                    |                                                                                                   |
|                                    | learning rate, loss funcion, metric and patience for early stopping                               |
+------------------------------------+---------------------------------------------------------------------------------------------------+
| metrics                            | Choose training metrics to monitor during training                                                |
+------------------------------------+---------------------------------------------------------------------------------------------------+

There are two ways to change the default configuration:

1. Update the parameter values inside yaml file for the respective configuration

2. Pass the argument in command line.

For the second option, user can pass the arguments explicitly stated in the main `config.yaml` file directly in the command line, such as follows:

.. code-block:: bash
   :linenos: 

   cd PeekdingDuck
   python ./peekingduck/training/main.py debug=True framework=tensorflow project_name=abcxyz view_only=True

To change the arguments in other configuration files such as `model`, `trainer`, etc., the user would need to chain up the arguments based on the hierarchy in the yaml files. Here is one example:

.. code-block:: bash
   :linenos: 

   cd PeekdingDuck
   python ./peekingduck/training/main.py debug=True framework=pytorch model.pytorch.model_name=mobilenetv3_small_050 trainer.pytorch.global_train_params.debug_epochs=5

Refer to the following sections to learn about the detailed configurations for customized training

.. _config-files-mainconfig:

Main Config
===========

Config File : ``peekingduck/training/configs/config.yaml``

.. raw:: html

   <div class="wy-table-responsive">
        <table class="docutils align-default">
            <thead>
                <tr class="row-odd">
                    <th colspan="3" class="head"><p>Key</p></th>
                    <th class="head"><p>Value</p></th>
                    <th class="head"><p>Description</p></th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td colspan="3"><code class="xref"><span class="pre">device</span></code></td>
                    <td><p>"auto"</p></td>
                    <td><p>"auto" to select available gpu. "cpu"|"cuda"|"mps"</p></td>
                </tr>
                <tr>
                    <td colspan="3"><code class="xref"><span class="pre">project_name</span></code></td>
                    <td><p>"cifar10"</p></td>
                    <td><p>rsna | cifar10 | vegfru5 | vegfru15 | vegfru25 | "your-project-name" </p></td>
                </tr>
                <tr>
                    <td colspan="3"><code class="xref"><span class="pre">debug</span></code></td>
                    <td><p>True</p></td>
                    <td><p>True | False</p></td>
                </tr>
                <tr>
                    <td colspan="3"><code class="xref"><span class="pre">framework</span></code></td>
                    <td><p>"tensorflow"</p></td>
                    <td><p>'pytorch' | 'tensorflow'</p></td>
                </tr>
                <tr>
                    <td colspan="3"><code class="xref"><span class="pre">random_state</span></code></td>
                    <td><p>11</p></td>
                    <td><p>Any number above 0</p></td>
                </tr>
                <tr>
                    <td colspan="3"><code class="xref"><span class="pre">view_only</span></code></td>
                    <td><p>False</p></td>
                    <td><p>True | False</p></td>
                </tr>
                <tr>
                    <td colspan="3"><code class="xref"><span class="pre">defaults</span></code></td>
                    <td><p></p></td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td colspan="2"><code class="xref"><span class="pre">use_case</span></code></td>
                    <td><p>classification</p></td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td colspan="2"><code class="xref"><span class="pre">data_module</span></code></td>
                    <td><p>cifar10</p></td>
                    <td><p>rsna | cifar10 | vegfru5 | vegfru15 | vegfru25 | "main-data_module-filename"</p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td colspan="2"><code><span class="pre grey">model</span></code></td>
                    <td><p class="grey">${use_case}</p></td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td colspan="2"><code><span class="pre grey">trainer</span></code></td>
                    <td><p class="grey">${use_case}</p></td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td colspan="2"><code><span class="pre grey">callbacks</span></code></td>
                    <td><p class="grey">${use_case}</p></td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td colspan="2"><code><span class="pre grey">metrics</span></code></td>
                    <td><p class="grey">${use_case}</p></td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td colspan="2"><code><span class="pre grey">model_analysis</span></code></td>
                    <td><p class="grey">${use_case}</p></td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td colspan="2"><code><span class="pre grey">stores</span></code></td>
                    <td><p class="grey">${use_case}</p></td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td colspan="2"><code><span class="pre grey">override hydra/job_logging</span></code></td>
                    <td><p class="grey">custom</p></td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td colspan="3"><code><span class="pre grey">hydra</span></code></td>
                    <td><p></p></td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td colspan="2"><code><span class="pre grey">run</span></code></td>
                    <td><p></p></td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td><p></p></td>
                    <td><code><span class="pre grey">dir</span></code></td>
                    <td><p class="grey">"outputs/${project_name}/${stores.unique_id}"</p></td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td colspan="2"><code><span class="pre grey">sweep</span></code></td>
                    <td><p></p></td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td><p></p></td>
                    <td><code><span class="pre grey">dir</span></code></td>
                    <td><p class="grey">outputs/${project_name}/${stores.unique_id}</p></td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><p></p></td>
                    <td><p></p></td>
                    <td><code><span class="pre grey">subdir</span></code></td>
                    <td><p class="grey">${hydra.job.num}</p></td>
                    <td><p></p></td>
                </tr>
            </tbody>
        </table>
   </div>

.. _config-files-datamodule:

Data Module
===========


.. raw:: html

   <h4>Resample / Train Test Split</h4>
   <p>Split arrays or matrices into random train and test subsets.</p>

Config File : ``peekingduck/training/configs/data_module/resample/train_test_split.yaml``

.. raw:: html

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
                    <td><code><span class="pre grey">random_state</span></code></td>
                    <td><p class="grey">${random_state}</p></td>
                    <td><p>Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.</p></td>
                </tr>

            </tbody>
        </table>
   </div>

.. raw:: html

   <div class="install">
     <strong>Select Framework : </strong>
     <button class="training-framework-btn training-framework-py active">Pytorch</button>
     <button class="training-framework-btn training-framework-tf">Tensorflow</button>
   </div>
   <br>

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

Config File : ``peekingduck/training/configs/data_module/dataset/``

.. tabs::

   .. tab:: Cifar10

      | https://www.cs.toronto.edu/~kriz/cifar.html. The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

      | You can download the dataset here : `dataset/cifar-10`_
      | You can download the labels csv file here : `csv/cifar-10`_

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

      | You can download the dataset here : `dataset/vegfru5`_ | `dataset/vegfru15`_ | `dataset/vegfru25`_
      | You can download the labels csv file here : `csv/vegfru5`_ | `csv/vegfru15`_ | `csv/vegfru25`_
      

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

      | You can download the dataset here : `dataset/rsna`_
      | You can download the labels csv file here : `csv/rsna`_

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



.. _`dataset/cifar-10`: https://www.cs.toronto.edu/~kriz/cifar.html#download
.. _`csv/cifar-10`: https://raw.githubusercontent.com/aisingapore/PeekingDuck/feat-training/data/cifar10/train.csv

.. _`dataset/vegfru5`: https://github.com/ustc-vim/vegfru#VegFru
.. _`csv/vegfru5`: https://github.com/aisingapore/PeekingDuck/blob/feat-training/data/vegfru/vegfru5.csv
.. _`dataset/vegfru15`: https://github.com/ustc-vim/vegfru#VegFru
.. _`csv/vegfru15`: https://github.com/aisingapore/PeekingDuck/blob/feat-training/data/vegfru/vegfru15.csv
.. _`dataset/vegfru25`: https://github.com/ustc-vim/vegfru#VegFru
.. _`csv/vegfru25`: https://github.com/aisingapore/PeekingDuck/blob/feat-training/data/vegfru/vegfru25.csv

.. _`dataset/rsna`: https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data
.. _`csv/rsna`: https://raw.githubusercontent.com/aisingapore/PeekingDuck/feat-training/data/rsna/train.csv

.. _config-files-model:

Model
=====

.. raw:: html

   <div class="install">
     <strong>Select Framework : </strong>
     <button class="training-framework-btn training-framework-py active">Pytorch</button>
     <button class="training-framework-btn training-framework-tf">Tensorflow</button>
   </div>
   <br>

.. raw:: html

   <div class="install">
     <div class="pkd-expandable training-pytorch">
        <h4>Pytorch</h4>
        <p>Config File : <code class="docutils literal notranslate"><span class="pre">peekingduck/training/configs/model/classification.yaml</span></code></p>
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
                        <td><p>"timm"</p></td>
                        <td><p>
                            PeekingDuck supports pre-trained models from both <code>torchvision</code> and <code>timm</code> libraries.
                            <br>
                            <br>Refer to the FAQ section for choosing between the two adapters for pre-trained models
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td colspan="2"><code><span class="pre grey">task</span></code></td>
                        <td><p class="grey">${use_case}</p></td>
                        <td>
                            Should not be changed. For hydra interpolation.
                        </td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">model_name</span></code></td>
                        <td><p>"vgg16"</p></td>
                        <td><p>
                            Supported model names from <code>torchvision</code> or <code>timm</code>.
                            <br>
                            <br>For <code>torchvision</code>, refer to the <a href="https://pytorch.org/vision/stable/models.html#classification">official docs</a> for the list of the model.
                            Click on the links and use the class name as the model name in the configuration file.
                            <br>
                            <br>For example, to use a vgg16 model, click the <a href="https://pytorch.org/vision/stable/models/vgg.html">VGG link</a> and use <code>vgg16</code> as the value for the model name.
                            <br>
                            <br>For <code>timm</code>, use <code>timm.list_models(pretrained=True)</code> to list out all the supported models, and use the string value as the value for the model name. 
                            <br>
                            <br>Refer to the <a href="https://timm.fast.ai/#List-Models-with-Pretrained-Weights">official docs</a> for advanced searching.
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">weights</span></code></td>
                        <td><p>"DEFAULT"</p></td>
                        <td>
                            </p>
                            Only applicable for <code>torchvision</code> adapter.
                            <br>
                            <br>Set as <code>DEFAULT</code> (recommended) to use default pre-trained weights, or change to alternative pre-trained weights (if supported) described in the documentation for the specific model.
                            <br>
                            <br>Refer to <a href="https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html#torchvision.models.MobileNet_V3_Large_Weights">mobilenet_v3_large</a> for an example.
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">pretrained</span></code></td>
                        <td><p>True</p></td>
                        <td>
                            Only applicable for <code>timm</code> adapter.
                            <br>If set to <code>False</code>, the weights will be initialized randomly.
                        </td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">fine_tune</span></code></td>
                        <td><p>True</p></td>
                        <td>
                            <code>True</code> to fine-tune the model after training the classifier.
                            <br><code>False</code> to only train the classifier.
                        </td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">fine_tune_all</span></code></td>
                        <td><p>True</p></td>
                        <td>
                            <p>
                            Only applicable when <code>fine_tune</code> is set to <code>True</code>
                            <br><code>True</code> to fine-tune the model after training the classifier.
                            <br><code>False</code> to only train the classifier.
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">fine_tune_modules</span></code></td>
                        <td><p></p></td>
                        <td><p>
                            Only applicable when <code>fine_tune_all</code> is set to <code>False</code>
                            <br>
                            <br>Specify which block within the model to fine-tune, accessed by the name.
                            <br>
                            <br>The settings will depend on the selected model, since each model architecture names the sub-modules differently.
                            <br>
                            <br>The sub-keys are the name of the modules to fine-tune, which can be viewed in the model print-out when setting <code>view_only</code> to <code>True</code> in the main configuration file.
                            <br>
                            <br>The values represent the layer/sub-modules to fine-tune, which can be an integer or a list of string. For an integer "n", it sets the last "n" layer/sub-modules to fine-tune. For a list of string, it sets the layer/sub-modules to fine-tune by names.
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">features</span></code></td>
                        <td><p>7</p></td>
                        <td>
                            This is an example value. For "vgg16" model,
                            <br>it will set the last <code>7</code> layers within the <code>feature</code> module as trainable
                        </td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">pre_logits</span></code></td>
                        <td><p>[
                                "fc1",
                                "act1",
                                "drop",
                                "fc2",
                                "act2"
                            ]</p></td>
                        <td><p>
                            This is an example value. For "vgg16" model, 
                            <br>it will set the <code>
                                "fc1",
                                "act1",
                                "drop",
                                "fc2",
                                "act2"
                            </code> modules as trainable
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td colspan="2"><code><span class="pre grey">num_classes</span></code></td>
                        <td><p class="grey">${data_module.dataset.num_classes}</p></td>
                        <td><p>Should not be changed. For hydra interpolation.</p></td>
                    </tr>
                    <tr>
                        <td colspan="2"><code><span class="pre grey">device</span></code></td>
                        <td><p class="grey">${device}</p></td>
                        <td><p>Should not be changed. For hydra interpolation.</p></td>
                    </tr>
                </tbody>
            </table>
        </div>
     
     </div>
     <div class="pkd-expandable training-tensorflow">
        <h4>Tensorflow</h4>
        <p>Config File : <code class="docutils literal notranslate"><span class="pre">peekingduck/training/configs/model/classification.yaml</span></code></p>
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
                        <td><code><span class="pre grey">task</span></code></td>
                        <td><p class="grey">${use_case}</p></td>
                        <td>
                            <p>Should not be changed. For hydra interpolation.</p>
                        </td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">model_name</span></code></td>
                        <td><p>"VGG16"</p></td>
                        <td>
                            <p>
                            Supported model names from <code>tf.keras.applications</code>.
                            <br>Refer to the <a href="https://www.tensorflow.org/api_docs/python/tf/keras/applications#functions">official docs</a> for the list of the model.
                            <br>Use the name of the function as the string value for the model name.                        
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td><code><span class="pre grey">num_classes</span></code></td>
                        <td><p class="grey">${data_module.dataset.num_classes}</p></td>
                        <td>
                            <p>Should not be changed. For hydra interpolation.</p>
                        </td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">activation</span></code></td>
                        <td><p>"softmax"</p></td>
                        <td>
                            <p>Activation function for the last layer of the classifier. Should not be changed.</p>
                        </td>
                    </tr>
                    <tr>
                        <td><code><span class="pre grey">image_size</span></code></td>
                        <td><p class="grey">${data_module.dataset.image_size}</p></td>
                        <td>
                            <p>Should not be changed. For hydra interpolation.</p>
                        </td>
                    </tr>
                    <tr>
                        <td><code><span class="pre grey">device</span></code></td>
                        <td><p class="grey">${device}</p></td>
                        <td>
                            <p>Should not be changed. For hydra interpolation.</p>
                        </td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">pretrained</span></code></td>
                        <td><p>True</p></td>
                        <td>
                            <p>Whether to use the pre-trained weights.</p>
                        </td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">fine_tune</span></code></td>
                        <td><p>True</p></td>
                        <td>
                            <p>Control whether to fine-tune the model after training the classifier.</p>
                        </td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">fine_tune_all</span></code></td>
                        <td><p>True</p></td>
                        <td>
                            <p>
                            Only applicable when <code>fine_tune</code> is set to <code>True</code>
                            <br><code>True</code> to fine-tune the model after training the classifier.
                            <br><code>False</code> to only train the classifier.
                            </p>
                        </td>
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
                        <td><p>
                            Unfreeze the layers by layer names to fine-tune. The name of the layers can be viewed by the model print-out.
                            </p>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>

     </div>
   </div>

.. _config-files-modelanalysis:

Model Analysis
==============

Config File : ``peekingduck/training/configs/model_analysis/classification.yaml``

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
                    <td><code class="xref"><span class="pre">entity</span></code></td>
                    <td>"peekingduck"</td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><code><span class="pre grey">project</span></code></td>
                    <td class="grey">"${project_name}"</td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><code><span class="pre grey">run_name</span></code></td>
                    <td class="grey">"${stores.unique_id}"</td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><code><span class="pre grey">framework</span></code></td>
                    <td class="grey">"${framework}"</td>
                    <td><p></p></td>
                </tr>
                <tr>
                    <td><code><span class="pre grey">debug</span></code></td>
                    <td class="grey">${debug}</td>
                    <td><p></p></td>
                </tr>
            </tbody>
        </table>
   </div>

.. _config-files-trainer:

Trainer
=======

.. raw:: html

   <div class="install">
     <strong>Select Framework : </strong>
     <button class="training-framework-btn training-framework-py active">Pytorch</button>
     <button class="training-framework-btn training-framework-tf">Tensorflow</button>
   </div>
   <br>
   

.. raw:: html

   <div class="install">
      <div class="pkd-expandable training-pytorch">
        <h4>Pytorch</h4>
        <p>The trainer class will make use of these configs.</p>
        <p>Config File : <code class="docutils literal notranslate"><span class="pre">peekingduck/training/configs/trainer/classification.yaml</span></code></p>

.. include:: /training_pipeline/02_configuring_training_parameters/trainer/pytorch_config.rst

.. raw:: html

   </div>
      <div class="pkd-expandable training-tensorflow">
        <h4>Tensorflow</h4>
        <p>The trainer class will make use of these configs.</p>
        <p>Config File : <code class="docutils literal notranslate"><span class="pre">peekingduck/training/configs/trainer/classification.yaml</span></code></p>

.. include:: /training_pipeline/02_configuring_training_parameters/trainer/tensorflow_config.rst

.. raw:: html

   </div></div>

.. _config-files-metrics:

Metrics
=======

.. raw:: html

   <div class="install">
     <strong>Select Framework : </strong>
     <button class="training-framework-btn training-framework-py active">Pytorch</button>
     <button class="training-framework-btn training-framework-tf">Tensorflow</button>
   </div>
   <br>

.. raw:: html

   <div class="install">
     <div class="pkd-expandable training-pytorch">
        <h4>Pytorch</h4>
        <p>Refer to <a href="https://torchmetrics.readthedocs.io/en/stable/all-metrics.html">Torch Metrics</a> documentation for more metrics you can use and their details.</p>
        <p>Values listed here are taken from the torch metrics api. It is important to note that the values are case-sensitive.</p>
        <p>The table below shows the default metrics:</p>
        <p>Config file : <code class="docutils literal notranslate"><span class="pre">peekingduck/training/configs/metrics/classification.yaml</span></code></p>
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
                        <td>"micro": Sum statistics over all labels
                            <br>"macro": Calculate statistics for each label and average them
                            <br>"weighted": Calculates statistics for each label and computes weighted average using their support "none"
                            <br>None: Calculates statistic for each label and applies no reduction
                        </td>
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
                        <td>micro: Sum statistics over all labels
                            <br>macro: Calculate statistics for each label and average them
                            <br>weighted: Calculates statistics for each label and computes weighted average using their support "none"
                            <br>None: Calculates statistic for each label and applies no reduction
                        </td>
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
                        <td>micro: Sum statistics over all labels
                            <br>macro: Calculate statistics for each label and average them
                            <br>weighted: Calculates statistics for each label and computes weighted average using their support "none"
                            <br>None: Calculates statistic for each label and applies no reduction
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
     
     </div>
     <div class="pkd-expandable training-tensorflow">
        <h4>Tensorflow</h4>
        <p>Refer to the <a href="https://www.tensorflow.org/api_docs/python/tf/keras/metrics">TensorFlow v2 Metrics</a> documentation for more metrics you can use and their details.</p>
        <p>Values listed here are taken from the tensorflow keras metrics api. It is important to note that the values are case-sensitive.</p>
        <p>The table below shows the default metrics:</p>
        <p>Config file : <code class="docutils literal notranslate"><span class="pre">peekingduck/training/configs/metrics/classification.yaml</span></code></p>
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
                        <td class="grey">${data_module.dataset.num_classes}</td>
                        <td></td>
                    </tr>
                </tbody>
            </table>
        </div>

     </div>
   </div>

.. _config-files-callbacks:

Callbacks
=========

.. raw:: html

   <div class="install">
     <strong>Select Framework : </strong>
     <button class="training-framework-btn training-framework-py active">Pytorch</button>
     <button class="training-framework-btn training-framework-tf">Tensorflow</button>
   </div>
   <br>

.. raw:: html

   <div class="install">
     <div class="pkd-expandable training-pytorch">
        <h4>Pytorch</h4>
        <p>These are the only available callbacks currently. It is important to note that the callbacks are automatically sorted (as per the list sequence below) during initialization:
        <ol>
            <li>History (required)</li>
            <li>MetricMeter (required)</li>
            <li>ModelCheckpoint (required)</li>
            <li>Logger</li>
            <li>EarlyStopping</li>
        </ol>
        </p>
        <p>The table below shows the default values:</p>
        <p>Config File : <code class="docutils literal notranslate"><span class="pre">peekingduck/training/configs/callbacks/classification.yaml</span></code></p>
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
                        <td>Incharge of printing the train and validation loop metrics and summaries.</td>
                    </tr>
                    <tr class="row-odd">
                        <td colspan="2"><code class="xref"><span class="pre">EarlyStopping</span></code></td>
                        <td></td>
                        <td>Stop training when a monitored metric has stopped improving.</td>
                    </tr>
                    <tr class="row-even">
                        <td><p></p></td>
                        <td><code><span class="pre grey">patience</span></code></td>
                        <td class="grey">${trainer.pytorch.global_train_params.patience}</td>
                        <td>Number of epochs with no improvement after which training will be stopped.</td>
                    </tr>
                    <tr class="row-odd">
                        <td><p></p></td>
                        <td><code><span class="pre grey">monitor</span></code></td>
                        <td class="grey">${trainer.pytorch.global_train_params.monitored_metric.monitor}</td>
                        <td>Name of the metric to monitor, should be one of the keys in metrics list.</td>
                    </tr>
                    <tr class="row-even">
                        <td><p></p></td>
                        <td><code><span class="pre grey">mode</span></code></td>
                        <td class="grey">${trainer.pytorch.global_train_params.monitored_metric.mode}</td>
                        <td>"min" or "max"
                        <br><br>In min mode, training will stop when the quantity monitored has stopped decreasing.
                        <br>In "max" mode it will stop when the quantity monitored has stopped increasing.</td>
                    </tr>
                    <tr class="row-even">
                        <td colspan="2"><code class="xref"><span class="pre">History</span></code></td>
                        <td></td>
                        <td>Callback that records events into a History object.</td>
                    </tr>
                    <tr class="row-odd">
                        <td colspan="2"><code class="xref"><span class="pre">MetricMeter</span></code></td>
                        <td></td>
                        <td>Calculates the cumulative metric score, cumulative count and average score
                        <br><br>
                        Note after 1 full loop epoch,
                        <br>the model has traversed through all batches in the dataloader.
                        <br>So, the average score is the average of all batches in the dataloader.
                        <br>for eg, if train set has 1000 samples and batch size is 100,
                        <br>then the model will have traversed through 10 batches in 1 epoch.
                        <br>then the cumulative count is "step" which is 10 in this case.
                        <br>the cumulative metric score is the sum of all the metric scores of all batches.
                        <br>so add up all the metric scores of all batches and divide by the cumulative count.
                        <br>this is the average score of all batches in 1 epoch.
                        </td>
                    </tr>
                    <tr class="row-even">
                        <td colspan="2"><code class="xref"><span class="pre">ModelCheckpoint</span></code></td>
                        <td></td>
                        <td>Callback to save the model or model weights at some frequency.</td>
                    </tr>
                    <tr class="row-even">
                        <td><p></p></td>
                        <td><code><span class="pre grey">monitor</span></code></td>
                        <td class="grey">${trainer.pytorch.global_train_params.monitored_metric.monitor}</td>
                        <td>Name of the metric to monitor, should be one of the keys in metrics list.</td>
                    </tr>
                    <tr class="row-odd">
                        <td><p></p></td>
                        <td><code><span class="pre grey">mode</span></code></td>
                        <td class="grey">${trainer.pytorch.global_train_params.monitored_metric.mode}</td>
                        <td>
                            "max" or "min"<br>
                            <br>In "min" mode, training will stop when the quantity monitored has stopped decreasing
                            <br>in "max" mode it will stop when the quantity monitored has stopped increasing.                        
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
     
     </div>
     <div class="pkd-expandable training-tensorflow">
        <h4>Tensorflow</h4>
        <p>Check out the <a href="https://www.tensorflow.org/api_docs/python/tf/keras/callbacks">TensorFlow v2 Callbacks</a> documentation for more details.
        <br>While technically you can use any callbacks listed in the keras API, only EarlyStopping has been tested.</p>
        <p>The table below shows the default values:</p>
        <p>Config File : <code class="docutils literal notranslate"><span class="pre">peekingduck/training/configs/callbacks/classification.yaml</span></code></p>
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
                        <td>Stop training when a monitored metric has stopped improving.</td>
                    </tr>
                    <tr class="row-odd">
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">patience</span></code></td>
                        <td>3</td>
                        <td>Number of epochs with no improvement after which training will be stopped.</td>
                    </tr>
                    <tr class="row-even">
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">restore_best_weights</span></code></td>
                        <td>True</td>
                        <td>Whether to restore model weights from the epoch with the best value of the monitored quantity.<br>If False, the model weights obtained at the last step of training are used.<br>An epoch will be restored regardless of the performance relative to the baseline.<br>If no epoch improves on baseline, training will run for patience epochs and restore weights from the best epoch in that set.</td>
                    </tr>
                    <tr class="row-odd">
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">monitor</span></code></td>
                        <td>"val_categorical_accuracy"</td>
                        <td>Metric to be monitored.</td>
                    </tr>

                    <tr class="row-even">
                        <td colspan="2"><code class="xref"><span class="pre">ProgbarLogger</span></code></td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr class="row-odd">
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">count_mode</span></code></td>
                        <td>'steps'</td>
                        <td></td>
                    </tr>
                    <tr class="row-even">
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">stateful_metrics</span></code></td>
                        <td>Null</td>
                        <td></td>
                    </tr>

                    <tr class="row-odd">
                        <td colspan="2"><code class="xref"><span class="pre">ModelCheckpoint</span></code></td>
                        <td></td>
                        <td>Save model checkpoints based on specified save_freq parameter frequency.</td>
                    </tr>
                    <tr class="row-even">
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre grey">filepath</span></code></td>
                        <td class="grey">${trainer.tensorflow.stores.model_artifacts_dir}</td>
                        <td></td>
                    </tr>
                    <tr class="row-odd">
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre grey">monitor</span></code></td>
                        <td class="grey">${trainer.tensorflow.global_train_params.monitored_metric.monitor}</td>
                        <td></td>
                    </tr>
                    <tr class="row-even">
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">save_best_only</span></code></td>
                        <td>False</td>
                        <td></td>
                    </tr>
                    <tr class="row-odd">
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">save_weights_only</span></code></td>
                        <td>False</td>
                        <td></td>
                    </tr>
                    <tr class="row-even">
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">mode</span></code></td>
                        <td>'auto'</td>
                        <td></td>
                    </tr>
                    <tr class="row-odd">
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">save_freq</span></code></td>
                        <td>'epoch'</td>
                        <td></td>
                    </tr>
                </tbody>
            </table>
        </div>

     </div>
   </div>

.. _config-files-store:

Store
=====

Config File : ``peekingduck/training/configs/stores/classifications.yaml``

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
                <tr class="row-even">
                    <td><code><span class="pre grey">unique_id</span></code></td>
                    <td><p class="grey">${now:%Y%m%d_%H%M%S}</p></td>
                    <td>Used in:<ul>
                        <li>- Hydra run and sweep directory path</li>
                        <li>- Model analysis run_name</li>
                        <li>- Pytorch Training model artifacts directory</li>
                    </ul></td>
                </tr>
            </tbody>
        </table>
   </div>