.. tabs::

   .. tab:: General 

      .. raw:: html 
      
         <p>This is a sentence describing Optimizer</p>
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
                        <td colspan="2"><code class="xref"><span class="pre">global_train_params</span></code></td>
                        <td><p></p></td>
                        <td><p>Global trainer parameters</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">manual_seed</span></code></td>
                        <td><p>${random_state}</p></td>
                        <td><p>Random seed. Default value will reference directly from main config file.</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">epochs</span></code></td>
                        <td><p>10</p></td>
                        <td><p>Number of epochs to train</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">debug</span></code></td>
                        <td><p>${debug}</p></td>
                        <td><p>Flag for checking if debug is set to True of False. Reference directly from main config file.</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">debug_epochs</span></code></td>
                        <td><p>3</p></td>
                        <td><p>When debug is set to True, this value will be used for training.</p></td>
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
                        <th colspan="3" class="head"><p>Key</p></th>
                        <th class="head"><p>Value</p></th>
                        <th class="head"><p>Description</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td colspan="3"><code class="xref"><span class="pre">optimizer_params</span></code></td>
                        <td><p></p></td>
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
                        <td><code class="xref"><span class="pre">beta_1</span></code></td>
                        <td><p>0.9</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">beta_2</span></code></td>
                        <td><p>0.999</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">epsilon</span></code></td>
                        <td><p>0.0000007</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">name</span></code></td>
                        <td><p>"Adam"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">finetune_learning_rate</span></code></td>
                        <td><p>1e-5</p></td>
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
                        <th colspan="3" class="head"><p>Key</p></th>
                        <th class="head"><p>Value</p></th>
                        <th class="head"><p>Description</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td colspan="3"><code class="xref"><span class="pre">lr_schedule_params</span></code></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">schedule</span></code></td>
                        <td><p>null</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">schedule_params</span></code></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">learning_rate</span></code></td>
                        <td><p>0.00001</p></td>
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
                        <td colspan="3"><code class="xref"><span class="pre">loss_params</span></code></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">loss_func</span></code></td>
                        <td><p>"CategoricalCrossentropy"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">loss_params</span></code></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">from_logits</span></code></td>
                        <td><p>False</p></td>
                        <td><p></p></td>
                    </tr>
                </tbody>
            </table>
         </div>