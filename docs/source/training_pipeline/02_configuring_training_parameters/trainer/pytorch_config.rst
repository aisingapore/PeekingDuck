
| The trainer class will make use of these configs.
| Config File : ``peekingduck/training/configs/trainer/classification.yaml``


.. tabs::

   .. tab:: General
      
      Global trainer parameters.

        +--------------------------------------------------------------------------------------------+--------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
        | Key                                                                                        | Value                                                  | Description                                                                                                                                        |
        +============================================================================================+========================================================+====================================================================================================================================================+
        | :mod:`global_train_params`                                                                 |                                                        |                                                                                                                                                    |
        +------------------------------------+-------------------------------------------------------+--------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                    | :grey:`manual_seed`                                   | :grey:`${random_state}`                                | Random seed. Default value will reference directly from main config file.                                                                          |
        +------------------------------------+-------------------------------------------------------+--------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                    | :mod:`epochs`                                         | 10                                                     | Number of epochs to train                                                                                                                          |
        +------------------------------------+-------------------------------------------------------+--------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                    | :mod:`patience`                                       | 3                                                      | Main reference value for early stopping patience count.                                                                                            |
        +------------------------------------+-------------------------------------------------------+--------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                    | :grey:`model_name`                                    | :grey:`${model.pytorch.model_name}`                    | Use for printing to logs. Default value will reference directly from model config.                                                                 |
        +------------------------------------+-------------------------------------------------------+--------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                    | :grey:`debug`                                         | :grey:`${debug}`                                       | Flag for checking if debug is set to True of False. Reference directly from main config file.                                                      |
        +------------------------------------+-------------------------------------------------------+--------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                    | :mod:`debug_epochs`                                   | 3                                                      | When debug is set to True, this value will be used for training.                                                                                   |
        +------------------------------------+-------------------------------------------------------+--------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                    | :grey:`classification_type`                           | :grey:`${data_module.dataset.classification_type}`     | Used in initializing PyTorch metrics. Values can be: 'binary', 'multiclass' or 'multilabel'. Default value references from dataset config file.    |
        +------------------------------------+-------------------------------------------------------+--------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                    | :mod:`monitored_metric`                               |                                                        |                                                                                                                                                    |
        +------------------------------------+---------------------------------+---------------------+--------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                    |                                 | :mod:`monitor`      | val_MulticlassAccuracy                                 | The metric used for monitoring the best validation score. This should be one of the keys in metrics list with a 'val_' prefix.                     |
        +------------------------------------+---------------------------------+---------------------+--------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                    |                                 | :mod:`mode`         | max                                                    | "min" | "max"                                                                                                                                      |
        |                                    |                                 |                     |                                                        |                                                                                                                                                    |
        |                                    |                                 |                     |                                                        | In min mode, training will stop when the quantity monitored has stopped decreasing.                                                                |
        |                                    |                                 |                     |                                                        |                                                                                                                                                    |
        |                                    |                                 |                     |                                                        | In ""max"" mode it will stop when the quantity monitored has stopped increasing."                                                                  |
        +------------------------------------+---------------------------------+---------------------+--------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+


   .. tab:: Optimizer

      These parameters will be used for initializing the optimizer. Implemented using the `torch.optim <https://pytorch.org/docs/stable/optim.html#algorithms>`_ package. Refer to `this documentation <https://pytorch.org/docs/stable/optim.html#algorithms>`_ for alternatives. Below is the default values using the Adam optimizer.

        +-----------------------------------------------------------------------------+-----------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
        | Key                                                                         | Value           | Description                                                                                                                                        |
        +=============================================================================+=================+====================================================================================================================================================+
        | :mod:`optimizer_params`                                                     |                 |                                                                                                                                                    |
        +-----------------------------+-----------------------------------------------+-----------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
        |                             | :mod:`optimizer`                              | "Adam"          | For further details regarding the algorithm we refer to `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_.           |
        +-----------------------------+-----------------------------------------------+-----------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
        |                             | :mod:`optimizer_params`                       |                 |                                                                                                                                                    |
        +-----------------------------+-----------------------------+-----------------+-----------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
        |                             |                             | :mod:`lr`       | 1.00E-05        | (float, optional) – learning rate (default: 1e-5).                                                                                                 |
        +-----------------------------+-----------------------------+-----------------+-----------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
        |                             |                             | :mod:`betas`    | [0.9, 0.999]    | (Tuple[float, float], optional) – coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))              |
        +-----------------------------+-----------------------------+-----------------+-----------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
        |                             |                             | :mod:`amsgrad`  | FALSE           | (bool, optional) – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond (default: False)      |
        +-----------------------------+-----------------------------+-----------------+-----------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
        |                             |                             | :mod:`eps`      | 0.0000007       | (float, optional) – term added to the denominator to improve numerical stability (default: 7e-7)                                                   |
        +-----------------------------+-----------------------------+-----------------+-----------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
        |                             | :mod:`finetune_params`                        |                 |                                                                                                                                                    |
        +-----------------------------+-----------------------------+-----------------+-----------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
        |                             |                             | :mod:`lr`       | 1.00E-05        | (float, optional) – learning rate.                                                                                                                 |
        +-----------------------------+-----------------------------+-----------------+-----------------+----------------------------------------------------------------------------------------------------------------------------------------------------+


   .. tab:: Scheduler

      .. raw:: html 

         <p>These parameters will be used for initializing the scheduler. Implemented using the <a target="_blank" href="https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate">torch.optim.lr_scheduler</a> package. Refer to <a target="_blank" href="https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate">PyTorch Optimizer Learning Rate</a> for scheduler choices. Scheduler is defaulted to null.</p>
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
                        <td><p></p></td>
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

      .. raw:: html 

         <p>Any of the scheduler listed in the <a target="_blank" href="https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate">torch.optim.lr_scheduler</a> method can be used. Below is an example using a OneCycleLR scheduler:</p>
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
                    <tr class="row-even">
                        <td colspan="3"><code class="xref"><span class="pre">scheduler_params</span></code></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr class="row-even">
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">scheduler</span></code></td>
                        <td><p>"OneCycleLR"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr class="row-even">
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">scheduler_params</span></code></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr class="row-even">
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">max_lr</span></code></td>
                        <td><p>1e-3</p></td>
                        <td><p></p></td>
                    </tr>
                </tbody>
            </table>
         </div>

         <p>You can also read <a target="_blank" href="https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863">this article</a> to find a suitable scheduler for your use case.</p>

   .. tab:: Loss

      .. raw:: html 

         <p>Refer to <a target="_blank" href="https://pytorch.org/docs/stable/nn.html#loss-functions">PyTorch Loss Functions</a> for more details and alternatives. The training pipeline defaults to using CrossEntropyLoss as the loss function for both train and validation loops.</p>
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
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><p><code class="xref"><span class="pre">train_criterion</span></code></p></td>
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
                        <td><p>(Tensor, optional) – a manual rescaling weight given to each class. If given, has to be a Tensor of size C.</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">size_average</span></code></p></td>
                        <td><p>null</p></td>
                        <td><p>(bool, optional) – Deprecated (see reduction). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field size_average is set to False, the losses are instead summed for each mini-batch. Ignored when reduce is False. Default: True</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">ignore_index</span></code></p></td>
                        <td><p>-100</p></td>
                        <td><p>(int, optional) – Specifies a target value that is ignored and does not contribute to the input gradient. When size_average is True, the loss is averaged over non-ignored targets. Note that ignore_index is only applicable when the target contains class indices.</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">reduce</span></code></p></td>
                        <td><p>null</p></td>
                        <td><p>(bool, optional) – Deprecated (see reduction). By default, the losses are averaged or summed over observations for each mini-batch depending on size_average. When reduce is False, returns a loss per batch element instead and ignores size_average. Default: True</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">reduction</span></code></p></td>
                        <td><p>"mean"</p></td>
                        <td><p>(str, optional) – Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the weighted mean of the output is taken, 'sum': the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: 'mean'</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">label_smoothing</span></code></p></td>
                        <td><p>0.0</p></td>
                        <td><p>(float, optional) – A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets become a mixture of the original ground truth and a uniform distribution as described in Rethinking the Inception Architecture for Computer Vision. Default: 0.0.</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><p><code class="xref"><span class="pre">valid_criterion</span></code></p></td>
                        <td><p>"CrossEntropyLoss"</p></td>
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
                        <td><p>(Tensor, optional) – a manual rescaling weight given to each class. If given, has to be a Tensor of size C.</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">size_average</span></code></p></td>
                        <td><p>null</p></td>
                        <td><p>(bool, optional) – Deprecated (see reduction). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field size_average is set to False, the losses are instead summed for each mini-batch. Ignored when reduce is False. Default: True</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">ignore_index</span></code></p></td>
                        <td><p>-100</p></td>
                        <td><p>(int, optional) – Specifies a target value that is ignored and does not contribute to the input gradient. When size_average is True, the loss is averaged over non-ignored targets. Note that ignore_index is only applicable when the target contains class indices.</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">reduce</span></code></p></td>
                        <td><p>null</p></td>
                        <td><p>(bool, optional) – Deprecated (see reduction). By default, the losses are averaged or summed over observations for each mini-batch depending on size_average. When reduce is False, returns a loss per batch element instead and ignores size_average. Default: True</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">reduction</span></code></p></td>
                        <td><p>"mean"</p></td>
                        <td><p>(str, optional) – Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the weighted mean of the output is taken, 'sum': the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: 'mean'</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><p><code class="xref"><span class="pre">label_smoothing</span></code></p></td>
                        <td><p>0.0</p></td>
                        <td><p>(float, optional) – A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets become a mixture of the original ground truth and a uniform distribution as described in Rethinking the Inception Architecture for Computer Vision. Default: 0.0.</p></td>
                    </tr>
                </tbody>
            </table>
         </div>


   .. tab:: Stores

      .. raw:: html 

         <p>Config used in saving model artifacts.</p>
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
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code><span class="pre grey">project_name</span></code></td>
                        <td><p class="grey">${project_name}</p></td>
                        <td><p>For used in model artifacts directory. Reference from main config file.</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code><span class="pre grey">unique_id</span></code></td>
                        <td><p class="grey">${stores.unique_id}</p></td>
                        <td><p>For used in model artifacts directory. Reference from store config file.</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code><span class="pre grey">model_artifacts_dir</span></code></td>
                        <td><p class="grey">"./outputs/${project_name}/${stores.unique_id}/"</p></td>
                        <td><p>The path to store the model artifacts.</p></td>
                    </tr>
                </tbody>
            </table>
         </div>
