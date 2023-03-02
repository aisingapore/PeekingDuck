.. tabs::

   .. tab:: General

      .. raw:: html 

         <p>This is a sentence describing Store</p>
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
                        <td><p><code class="xref"><span class="pre">manual_seed</span></code></p><td>
                        <td><p>${random_state}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">epochs</span></code></p><td>
                        <td><p>10</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">patience</span></code></p><td>
                        <td><p>3</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">model_name</span></code></p><td>
                        <td><p>${model.pytorch.model_name}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">debug</span></code></p><td>
                        <td><p>${debug}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">debug_epochs</span></code></p><td>
                        <td><p>3</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">classification_type</span></code></p><td>
                        <td><p>${data_module.dataset.classification_type}</p
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">monitored_metric</span></code></p><td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">monitored_metric</span></code>.monitor</p><td>
                        <td><p>val_MulticlassAccuracy</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">monitored_metric</span></code>.mode</p><td>
                        <td><p>max</p></td>
                        <td><p></p></td>
                    </tr>
                </tbody>
            </table>
         </div>

   .. tab:: Stores

      .. raw:: html 

         <p>This is a sentence describing Store</p>
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
                        <td><p>project_name</p></td>
                        <td><p>${project_name}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p>unique_id</p></td>
                        <td><p>${stores.unique_id} # field(default_factory=generate_uuid4)</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p>logs_dir</p></td>
                        <td><p>"" # Path = field(init=False)</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p>model_artifacts_dir</p></td>
                        <td><p>"./outputs/${project_name}/${stores.unique_id}/"</p></td>
                        <td><p></p></td>
                    </tr>
                </tbody>
            </table>
         </div>

   .. tab:: Optimizer

      .. raw:: html 

         <p>This is a sentence describing Optimizer</p>
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
                        <td><p>optimizer</p></td>
                        <td><p>"Adam"</p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p>optimizer_params</p></td>
                        <td><p></p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p>lr</p></td>
                        <td><p>1e-5 # bs: 32 -> lr = 3e-4</p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p>betas</p></td>
                        <td><p>[0.9, 0.999]</p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p>amsgrad</p></td>
                        <td><p>False</p</td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p>eps</p></td>
                        <td><p>0.0000007</p</td>
                        <td><p></p></td>
                    </tr>
                </tbody>
            </table>
         </div>

   .. tab:: Scheduler

      .. raw:: html 

         <p>This is a sentence describing Scheduler</p>
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
                    <tr class="row-even"><td><p>scheduler</p></td>
                        <td><p>null</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr class="row-even"><td><p>scheduler_params</p></td>
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
                        <th class="head"><p>Key</p></th>
                        <th class="head"><p>Value</p></th>
                        <th class="head"><p>Description</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><p><code class="xref"><span class="pre">train_criterion</span></code></p></td>
                        <td><p>"CrossEntropyLoss"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">valid_criterion</span></code></p></td>
                        <td><p>"CrossEntropyLoss"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">train_criterion_params</span></code></p></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">train_criterion_params</span></code>.weight</p></td>
                        <td><p>null</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">train_criterion_params</span></code>.size_average</p></td>
                        <td><p>null</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">train_criterion_params</span></code>.ignore_index</p></td>
                        <td><p>-100</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">train_criterion_params</span></code>.reduce</p></td>
                        <td><p>null</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">train_criterion_params</span></code>.reduction</p></td>
                        <td><p>"mean"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">train_criterion_params</span></code>.label_smoothing</p></td>
                        <td><p>0.0</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">valid_criterion_params</span></code></p></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">valid_criterion_params</span></code>.weight</p></td>
                        <td><p>null</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">valid_criterion_params</span></code>.size_average</p></td>
                        <td><p>null</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">valid_criterion_params</span></code>.ignore_index</p></td>
                        <td><p>-100</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">valid_criterion_params</span></code>.reduce</p></td>
                        <td><p>null</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">valid_criterion_params</span></code>.reduction</p></td>
                        <td><p>"mean"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p><code class="xref"><span class="pre">valid_criterion_params</span></code>.label_smoothing</p></td>
                        <td><p>0.0</p></td>
                        <td><p></p></td>
                    </tr>
                </tbody>
            </table>
         </div>