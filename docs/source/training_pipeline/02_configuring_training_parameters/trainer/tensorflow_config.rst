.. tabs::

   .. tab:: General 

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
                        <td colspan="3"><code class="xref"><span class="pre">global_train_params</span></code></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">manual_seed</span></code></td>
                        <td><p>${random_state}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">epochs</span></code></td>
                        <td><p>10</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">debug</span></code></td>
                        <td><p>${debug}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">debug_epochs</span></code></td>
                        <td><p>3</p></td>
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
                        <td colspan="3"><code class="xref"><span class="pre">optimizer_params</span></code></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">optimizer</span></code></td>
                        <td><p>"Adam"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">optimizer_params</span></code></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">optimizer_params</span></code>.beta_1</td>
                        <td><p>0.9</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">optimizer_params</span></code>.beta_2</td>
                        <td><p>0.999</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">optimizer_params</span></code>.epsilon</td>
                        <td><p>0.0000007</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">optimizer_params</span></code>.name</td>
                        <td><p>"Adam"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">finetune_learning_rate</span></code></td>
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
                        <th class="head"><p>Key</p></th>
                        <th class="head"><p>Value</p></th>
                        <th class="head"><p>Description</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td colspan="3"><code class="xref"><span class="pre">lr_schedule_params</span></code></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">schedule</span></code></td>
                        <td><p>null</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">schedule_params</span></code></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">schedule_params</span></code>.learning_rate</td>
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
                        <th class="head"><p>Key</p></th>
                        <th class="head"><p>Value</p></th>
                        <th class="head"><p>Description</p></th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td colspan="3"><code class="xref"><span class="pre">loss_params</span></code></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">loss_func</span></code></td>
                        <td><p>"CategoricalCrossentropy"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">loss_params</span></code></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">loss_params</span></code>.from_logits</td>
                        <td><p>False</p></td>
                        <td><p></p></td>
                    </tr>
                </tbody>
            </table>
         </div>