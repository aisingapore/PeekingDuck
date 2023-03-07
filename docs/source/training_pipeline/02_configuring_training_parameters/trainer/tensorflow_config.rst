.. tabs::

   .. tab:: General 

      .. raw:: html 
      
         <p>Global trainer parameters</p>
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
                        <td><p></p></td>
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
         
         <p>These parameters will be used for initializing the optimizer. Implemented using the <a target="_blank" href="https://keras.io/api/optimizers/">tf.keras.optimizers</a> package. Refer to <a target="_blank" href="https://keras.io/api/optimizers/">this documentation</a> for alternatives. Below is the default values using the Adam optimizer.</p>
         <p>Note that the learning rate parameter is not specified here but will refer to the scheduler config file instead.</p>
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
                        <td><p>A float value or a constant float tensor, or a callable that takes no arguments and returns the actual value to use. The exponential decay rate for the 1st moment estimates. Defaults to 0.9.</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">beta_2</span></code></td>
                        <td><p>0.999</p></td>
                        <td><p>A float value or a constant float tensor, or a callable that takes no arguments and returns the actual value to use. The exponential decay rate for the 2nd moment estimates. Defaults to 0.999.</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">epsilon</span></code></td>
                        <td><p>0.0000007</p></td>
                        <td><p>epsilon: A small constant for numerical stability. This epsilon is "epsilon hat" in the Kingma and Ba paper (in the formula just before Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to 7e-7.</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">name</span></code></td>
                        <td><p>"Adam"</p></td>
                        <td><p>name: String. The name to use for momentum accumulator weights created by the optimizer.</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">finetune_learning_rate</span></code></td>
                        <td><p>1e-5</p></td>
                        <td><p>Learning rate used for reinitialising optimizer during fine tuning.</p></td>
                    </tr>
                </tbody>
            </table>
         </div>

   .. tab:: Scheduler

      .. raw:: html 

         <p>These parameters will be used for initializing the scheduler. Implemented using the <a target="_blank" href="https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules">tf.keras.optimizers.schedules</a> package. Refer to <a target="_blank" href="https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules">TensorFlow Learning Rate Schedule</a> for scheduler choices. Scheduler is defaulted to null with a base learning rate of 1e-5.</p>
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
                        <td><p>1e-5</p></td>
                        <td><p></p></td>
                    </tr>
                </tbody>
            </table>
         </div>

      .. raw:: html  

         <p>Any of the scheduler listed in the <a target="_blank" href="https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules">tf.keras.optimizers.schedules</a> package can be used. Below is an example using a CosineDecayRestarts scheduler:</p>
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
                        <td><p>"CosineDecayRestarts"</p></td>
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
                        <td><code class="xref"><span class="pre">initial_learning_rate</span></code></td>
                        <td><p>0.003</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">first_decay_steps</span></code></td>
                        <td><p>10</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">t_mul</span></code></td>
                        <td><p>2.0</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">m_mul</span></code></td>
                        <td><p>1.0</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">alpha</span></code></td>
                        <td><p>0.0</p></td>
                        <td><p></p></td>
                    </tr>
                </tbody>
            </table>
         </div>


   .. tab:: Loss

      .. raw:: html 

         <p>Refer to <a target="_blank" href="https://www.tensorflow.org/api_docs/python/tf/keras/losses">TensorFlow Loss Functions</a> for more details and alternatives. The training pipeline defaults to using CategoricalCrossentropy as the loss function.</p>
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
                        <td><p>Computes the crossentropy loss between the labels and predictions.</p></td>
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
                        <td><p>Whether y_pred is expected to be a logits tensor. By default, we assume that y_pred encodes a probability distribution.</p></td>
                    </tr>
                </tbody>
            </table>
         </div>