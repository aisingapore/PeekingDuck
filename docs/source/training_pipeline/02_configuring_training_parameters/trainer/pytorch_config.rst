.. tabs::

   .. tab:: General

      .. raw:: html 

         <p>Global trainer parameters.</p>
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
                        <td colspan="3"><p><code class="xref"><span class="pre">global_train_params</span></code></p></td>
                        <td><p>${random_state}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><p><code class="xref"><span class="pre">manual_seed</span></code></p></td>
                        <td><p>${random_state}</p></td>
                        <td><p>Random seed. Default value will reference directly from main config file.</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><p><code class="xref"><span class="pre">epochs</span></code></p></td>
                        <td><p>10</p></td>
                        <td><p>Number of epochs to train</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><p><code class="xref"><span class="pre">patience</span></code></p></td>
                        <td><p>3</p></td>
                        <td><p>Main reference value for early stopping patience count.</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><p><code class="xref"><span class="pre">model_name</span></code></p></td>
                        <td><p>${model.pytorch.model_name}</p></td>
                        <td><p>Use for printing to logs. Default value will reference directly from model config.</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><p><code class="xref"><span class="pre">debug</span></code></p></td>
                        <td><p>${debug}</p></td>
                        <td><p>Flag for checking if debug is set to True of False. Reference directly from main config file.</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><p><code class="xref"><span class="pre">debug_epochs</span></code></p></td>
                        <td><p>3</p></td>
                        <td><p>When debug is set to True, this value will be used for training.</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><p><code class="xref"><span class="pre">classification_type</span></code></p></td>
                        <td><p>${data_module.dataset.classification_type}</p></td>
                        <td><p>Used in initializing PyTorch metrics. Values can be: 'binary', 'multiclass' or 'multilabel'. Default value references from dataset config file.</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><p><code class="xref"><span class="pre">monitored_metric</span></code></p></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">monitor</span></code></p></td>
                        <td><p>val_MulticlassAccuracy</p></td>
                        <td><p>The metric used for monitoring the best validation score. This should be one of the keys in metrics list with a 'val_' prefix.</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">mode</span></code></p></td>
                        <td><p>max</p></td>
                        <td><p>"min" | "max"
                            <br>In min mode, training will stop when the quantity monitored has stopped decreasing.
                            <br>In "max" mode it will stop when the quantity monitored has stopped increasing.</p></td>
                    </tr>
                </tbody>
            </table>
         </div>


   .. tab:: Optimizer

      .. raw:: html 

         <p>These parameters will be used for initializing the optimizer.</p>
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
                        <td colspan="3"><code class="xref"><span class="pre">optimizer_params</span></code></td>
                        <td><p></p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">optimizer</span></code></td>
                        <td><p>"Adam"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">optimizer_params</span></code></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">lr</span></code></td>
                        <td><p>1e-5 # bs: 32 -> lr = 3e-4</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">betas</span></code></td>
                        <td><p>[0.9, 0.999]</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">amsgrad</span></code></td>
                        <td><p>False</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">eps</span></code></td>
                        <td><p>0.0000007</p></td>
                        <td><p></p></td>
                    </tr>
                </tbody>
            </table>
         </div>

   .. tab:: Scheduler

      .. raw:: html 

         <p>These parameters will be used for initializing the scheduler.</p>
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
                        <td colspan="2"><code class="xref"><span class="pre">scheduler_params</span></code></td>
                        <td><p></p></td>
                        <td><p>Scheduler parameters</p></td>
                    </tr>
                    <tr class="row-even">
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">scheduler</span></code></td>
                        <td><p>null</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr class="row-even">
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">scheduler_params</span></code></td>
                        <td><p>null</p></td>
                        <td><p></p></td>
                    </tr>
                </tbody>
            </table>
         </div>

   .. tab:: Loss

      .. raw:: html 

         <p>This is a sentence describing Loss</p>
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
                        <td colspan="3"><p><code class="xref"><span class="pre">criterion_params</span></code></p></td>
                        <td><p></p></td>
                        <td><p>Loss function parameters</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><p><code class="xref"><span class="pre">train_criterion</span></code></p></td>
                        <td><p>"CrossEntropyLoss"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><p><code class="xref"><span class="pre">valid_criterion</span></code></p></td>
                        <td><p>"CrossEntropyLoss"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><p><code class="xref"><span class="pre">train_criterion_params</span></code></p></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">weight</span></code></p></td>
                        <td><p>null</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">size_average</span></code></p></td>
                        <td><p>null</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">ignore_index</span></code></p></td>
                        <td><p>-100</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">reduce</span></code></p></td>
                        <td><p>null</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">reduction</span></code></p></td>
                        <td><p>"mean"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">label_smoothing</span></code></p></td>
                        <td><p>0.0</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><p><code class="xref"><span class="pre">valid_criterion_params</span></code></p></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">weight</span></code></p></td>
                        <td><p>null</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">size_average</span></code></p></td>
                        <td><p>null</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">ignore_index</span></code></p></td>
                        <td><p>-100</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">reduce</span></code></p></td>
                        <td><p>null</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">reduction</span></code></p></td>
                        <td><p>"mean"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">label_smoothing</span></code></p></td>
                        <td><p>0.0</p></td>
                        <td><p></p></td>
                    </tr>
                </tbody>
            </table>
         </div>


   .. tab:: Stores

      .. raw:: html 

         <p></p>
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
                        <td colspan="2"><code class="xref"><span class="pre">stores</span></code></td>
                        <td><p>${project_name}</p></td>
                        <td><p>Stores parameters</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">project_name</span></code></td>
                        <td><p>${project_name}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">unique_id</span></code></td>
                        <td><p>${stores.unique_id} # field(default_factory=generate_uuid4)</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">logs_dir</span></code></td>
                        <td><p>"" # Path = field(init=False)</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">model_artifacts_dir</span></code></td>
                        <td><p>"./outputs/${project_name}/${stores.unique_id}/"</p></td>
                        <td><p></p></td>
                    </tr>
                </tbody>
            </table>
         </div>