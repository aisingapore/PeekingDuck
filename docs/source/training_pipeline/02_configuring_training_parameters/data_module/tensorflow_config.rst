.. tabs::

   .. tab:: Adapter

      .. raw:: html 

         <p>Configuration for Data Adapter to pipeline the dataset to the model.</p>
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
                        <td colspan="3"><code class="xref"><span class="pre">adapter_type</span></code></td>
                        <td><p>"tensorflow"</p></td>
                        <td><p>framework</p></td>
                    </tr>
                    <tr>
                        <td colspan="3"><code class="xref"><span class="pre">train</span></code></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">batch_size</span></code></td>
                        <td><p>32</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">shuffle</span></code></td>
                        <td><p>True</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">x_col</span></code></td>
                        <td><p>${data_module.dataset.image_path_col_name}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">y_col</span></code></td>
                        <td><p>${data_module.dataset.target_col_id}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">target_size</span></code></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre"> - </span></code></td>
                        <td><p>${data_module.dataset.image_size}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre"> - </span></code></td>
                        <td><p>${data_module.dataset.image_size}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">color_mode</span></code></td>
                        <td><p>"rgb"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">num_classes</span></code></td>
                        <td><p>${data_module.dataset.num_classes}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">classes</span></code></td>
                        <td><p>${data_module.dataset.class_name_to_id}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">class_mode</span></code></td>
                        <td><p>"categorical"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">interpolation</span></code></td>
                        <td><p>"nearest"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">subset</span></code></td>
                        <td><p>"training"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td colspan="3"><code class="xref"><span class="pre">valid</span></code></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">batch_size</span></code></td>
                        <td><p>32</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">shuffle</span></code></td>
                        <td><p>False</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">x_col</span></code></td>
                        <td><p>${data_module.dataset.image_path_col_name}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">y_col</span></code></td>
                        <td><p>${data_module.dataset.target_col_id}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">target_size</span></code></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre"> - </span></code></td>
                        <td><p>${data_module.dataset.image_size}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre"> - </span></code></td>
                        <td><p>${data_module.dataset.image_size}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">color_mode</span></code></td>
                        <td><p>"rgb"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">num_classes</span></code></td>
                        <td><p>${data_module.dataset.num_classes}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">classes</span></code></td>
                        <td><p>${data_module.dataset.class_name_to_id}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">class_mode</span></code></td>
                        <td><p>"categorical"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">interpolation</span></code></td>
                        <td><p>"nearest"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">subset</span></code></td>
                        <td><p>"validation"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td colspan="3"><code class="xref"><span class="pre">test</span></code></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">batch_size</span></code></td>
                        <td><p>1</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">shuffle</span></code></td>
                        <td><p>False</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">x_col</span></code></td>
                        <td><p>${data_module.dataset.image_path_col_name}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">y_col</span></code></td>
                        <td><p>${data_module.dataset.target_col_id}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">target_size</span></code></td>
                        <td><p></p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre"> - </span></code></td>
                        <td><p>${data_module.dataset.image_size}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre"> - </span></code></td>
                        <td><p>${data_module.dataset.image_size}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">color_mode</span></code></td>
                        <td><p>"rgb"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">num_classes</span></code></td>
                        <td><p>${data_module.dataset.num_classes}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">classes</span></code></td>
                        <td><p>${data_module.dataset.class_name_to_id}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">class_mode</span></code></td>
                        <td><p>"categorical"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">interpolation</span></code></td>
                        <td><p>"nearest"</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td colspan="2"><code class="xref"><span class="pre">subset</span></code></td>
                        <td><p>"validation"</p></td>
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
                        <td colspan="2"><code class="xref"><span class="pre">_target_</span></code></td>
                        <td><p>albumentations.augmentations.crops.transforms.RandomResizedCrop</p></td>
                        <td><p>Torchvision's variant of crop a random part of the input and rescale it to some size.</p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">height</span></code></td>
                        <td><p>${data_module.dataset.image_size}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">width</span></code></td>
                        <td><p>${data_module.dataset.image_size}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">scale</span></code></td>
                        <td><p>[0.9, 1]</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">ratio</span></code></td>
                        <td><p>[1, 1]</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">_target_</span></code></td>
                        <td><p>src.transforms.augmentations.TFPreprocessImage</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">preprocessor</span></code></td>
                        <td><p>keras.applications.vgg16.preprocess_input</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">p</span></code></td>
                        <td><p>1.</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">_target_</span></code></td>
                        <td><p>albumentations.augmentations.geometric.transforms.Flip</p></td>
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
                        <td colspan="2"><code class="xref"><span class="pre">_target_</span></code></td>
                        <td><p>albumentations.augmentations.geometric.resize.Resize</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">height</span></code></td>
                        <td><p>${data_module.dataset.image_size}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">width</span></code></td>
                        <td><p>${data_module.dataset.image_size}</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td colspan="2"><code class="xref"><span class="pre">_target_</span></code></td>
                        <td><p>src.transforms.augmentations.TFPreprocessImage</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">preprocessor</span></code></td>
                        <td><p>keras.applications.vgg16.preprocess_input</p></td>
                        <td><p></p></td>
                    </tr>
                    <tr>
                        <td><p></p></td>
                        <td><code class="xref"><span class="pre">p</span></code></td>
                        <td><p>1.</p></td>
                        <td><p></p></td>
                    </tr>
                </tbody>
            </table>
         </div>
