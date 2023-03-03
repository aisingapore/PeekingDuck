.. tabs::

   .. tab:: Adapter

      .. raw:: html 

         <p>Configuration for Data Adapter to pipeline the dataset to the model.</p>
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
                        <td><code class="xref"><span class="pre">adapter_type</span></code></td>
                        <td><p>"pytorch"</p></td>
                        <td><p>framework</p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">train</span></code></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">train</span></code>.batch_size</td>
                        <td><p>32</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">train</span></code>.shuffle</td>
                        <td><p>True</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">train</span></code>.pin_memory</td>
                        <td><p>True</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">train</span></code>.drop_last</td>
                        <td><p>False</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">valid</span></code></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">valid</span></code>.batch_size</td>
                        <td><p>32</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">valid</span></code>.shuffle</td>
                        <td><p>False</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">valid</span></code>.pin_memory</td>
                        <td><p>True</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">valid</span></code>.drop_last</td>
                        <td><p>False</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">test</span></code></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">test</span></code>.batch_size</td>
                        <td><p>1</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">test</span></code>.shuffle</td>
                        <td><p>False</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">test</span></code>.pin_memory</td>
                        <td><p>True</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><code class="xref"><span class="pre">test</span></code>.drop_last</td>
                        <td><p>False</p></td>
                        <td><p></p></td>
                    </tr>
                </tbody>
            </table>
         </div>

   .. tab:: Transform

      .. raw:: html 

         <h5>Train</h5>
         <p>Transformation to be applied to training dataset.</p>
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
                        <td colspan="2"><span>_target_</span></td>
                        <td><p>albumentations.augmentations.crops.transforms.RandomResizedCrop</p></td>
                        <td><p>Torchvision's variant of crop a random part of the input and rescale it to some size.</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><span>height</span></td>
                        <td><p>${data_module.dataset.image_size}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><span>width</span></td>
                        <td><p>${data_module.dataset.image_size}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><span>scale</span></td>
                        <td><p>[0.9, 1]</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><span>ratio</span></td>
                        <td><p>[1, 1]</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td colspan="2"><span>_target_</span></td>
                        <td><p>albumentations.augmentations.geometric.transforms.Flip</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td colspan="2"><span>_target_</span></td>
                        <td><p>albumentations.augmentations.transforms.Normalize</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><span>mean</span></td>
                        <td><p>[0.4913997551666284, 0.48215855929893703, 0.4465309133731618]</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><span>std</span></td>
                        <td><p>[0.24703225141799082, 0.24348516474564, 0.26158783926049628]</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td colspan="2"><span>_target_</span></td>
                        <td><p>albumentations.pytorch.transforms.ToTensorV2</p></td>
                        <td><p></p></td>
                    </tr>
                </tbody>
            </table>
         </div>

         <h5>Test</h5>
         <p>Transformation to be applied to test and validation dataset.</p>
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
                        <td colspan="2"><span>_target_</span></td>
                        <td><p>albumentations.augmentations.geometric.resize.Resize</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><span>height</span></td>
                        <td><p>${data_module.dataset.image_size}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><span>width</span></td>
                        <td><p>${data_module.dataset.image_size}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td colspan="2"><span>_target_</span></td>
                        <td><p>albumentations.augmentations.transforms.Normalize</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><span>mean</span></td>
                        <td><p>[0.4913997551666284, 0.48215855929893703, 0.4465309133731618]</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><span>std</span></td>
                        <td><p>[0.24703225141799082, 0.24348516474564, 0.26158783926049628]</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td colspan="2"><span>_target_</span></td>
                        <td><p>albumentations.pytorch.transforms.ToTensorV2</p></td>
                        <td><p></p></td>
                    </tr>
                </tbody>
            </table>
         </div>
