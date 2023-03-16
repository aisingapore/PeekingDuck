:orphan:

.. _training_pipeline:

***********************
Training your own model
***********************

.. toctree::
   :maxdepth: 2

   /training_pipeline/01_getting_started
   /training_pipeline/02_configuring_training_parameters/index
   /training_pipeline/03_using_custom_dataset

Documentation Convention
========================

.. _documentation_convention:

| In the following sections, we will be talking about the training pipeline and also showing you how you can configure this tranining pipeline to run on your own dataset.
| As such, this will be multiple instances talking about the config files that you will interact with to change the pipeline settings.
| These config files will be displayed in a table format, illustrated in the following table.

| The text color scheme shown is used to illustrate which of the values can be changed, and the ones that should not:

.. raw:: html

   <div class="wy-table-responsive">
      <table class="docutils align-default">
         <thead>
         <tr class="row-odd">
            <th class="head"><p>Color</p></th>
            <th class="head"><p>Context</p></th>
         </tr>
         </thead>
         <tbody>
         <tr class="row-even">
            <td><p class="grey">Grey</p></td>
            <td><p class="grey">Required Value (Don't change this)`</p></td>
         </tr>
         <tr class="row-odd">
            <td><p>Black</p></td>
            <td><p>This value can be changed</p></td>
         </tr>
         </tbody>
      </table>
   </div>


Example
*******

.. raw:: html

   <style>
      .grey {
         color: #c2c2c2;
      }
   </style>

   <div class="wy-table-responsive">
      <table class="docutils align-default">
            <thead>
               <tr class="row-odd">
                  <th colspan="2" class="head"><p>Key</p></th>
                  <th class="head"><p>Value</p></th>
                  <th class="head"><p>Action</p></th>
               </tr>
            </thead>
            <tbody>
               <tr>
                  <td colspan="2"><code class="xref"><span class="pre">adapter</span></code></td>
                  <td><p>"timm"</p></td>
                  <td><p>Can be changed.</p></td>
               </tr>
               <tr>
                  <td colspan="2"><code><span class="pre grey">task</span></code></td>
                  <td><p class="grey">${use_case}</p></td>
                  <td><p class="grey">Should not be changed.</p></td>
               </tr>
               <tr>
                  <td colspan="2"><code class="xref"><span class="pre">model_name</span></code></td>
                  <td><p>"vgg16"</p></td>
                  <td><p>Can be changed.</p></td>
               </tr>
               <tr>
                  <td colspan="2"><code class="xref"><span class="pre">weights</span></code></td>
                  <td><p>"DEFAULT"</p></td>
                  <td><p>Can be changed.</p></td>
               </tr>
               <tr>
                  <td colspan="2"><code class="xref"><span class="pre">pretrained</span></code></td>
                  <td><p>True</p></td>
                  <td><p>Can be changed.</p></td>
               </tr>
               <tr>
                  <td colspan="2"><code class="xref"><span class="pre">fine_tune</span></code></td>
                  <td><p>True</p></td>
                  <td><p>Can be changed.</p></td>
               </tr>
               <tr>
                  <td colspan="2"><code class="xref"><span class="pre">fine_tune_all</span></code></td>
                  <td><p>True</p></td>
                  <td><p>Can be changed.</p></td>
               </tr>
               <tr>
                  <td colspan="2"><code class="xref"><span class="pre">fine_tune_modules</span></code></td>
                  <td><p></p></td>
                  <td><p>Can be changed.</p></td>
               </tr>
               <tr>
                  <td><p></p></td>
                  <td><code class="xref"><span class="pre">features</span></code></td>
                  <td><p>7</p></td>
                  <td><p>Can be changed.</p></td>
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
                  <td><p>Can be changed.</p></td>
               </tr>
               <tr>
                  <td colspan="2"><code><span class="pre grey">num_classes</span></code></td>
                  <td><p class="grey">${data_module.dataset.num_classes}</p></td>
                  <td><p class="grey">Should not be changed.</p></td>
               </tr>
               <tr>
                  <td colspan="2"><code><span class="pre grey">device</span></code></td>
                  <td><p class="grey">${device}</p></td>
                  <td><p class="grey">Should not be changed.</p></td>
               </tr>
            </tbody>
      </table>
   </div>