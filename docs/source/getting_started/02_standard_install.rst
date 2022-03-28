****************
Standard Install
****************

.. include:: /include/substitution.rst


Install PeekingDuck
===================

.. raw:: html

   <div class="install">
     <strong>Operating System</strong>
     <input type="radio" name="os" id="quickstart-win" checked>
     <label for="quickstart-win">Windows</label>
     <input type="radio" name="os" id="quickstart-mac">
     <label for="quickstart-mac">macOS</label>
     <input type="radio" name="os" id="quickstart-lin">
     <label for="quickstart-lin">Linux</label><br />

     <strong>Virtual Environment</strong>
     <input type="radio" name="packager" id="quickstart-cond" checked>
     <label for="quickstart-cond">conda</label>
     <input type="radio" name="packager" id="quickstart-venv">
     <label for="quickstart-venv">venv</label>
     <input type="radio" name="packager" id="quickstart-none">
     <label for="quickstart-none">None</label>

     <div>
       <span class="pkd-expandable" data-venv="conda">
         Install conda using the
         <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/install/">Anaconda or miniconda</a>
         installers or the <a href="https://github.com/conda-forge/miniforge#miniforge3">miniforge</a>
         installers (recommended).
       </span>
       <span class="pkd-expandable" data-venv="venv" data-os="windows">
         Install Python 3 (64-bit) from <a href="https://www.python.org/">https://www.python.org</a>.
       </span>
       <span class="pkd-expandable" data-venv="venv" data-os="mac">
         Install Python 3 using <a href="https://brew.sh/">homebrew</a>
         (<code class="literal">brew install python</code>) or by manually installing the package from
         <a href="https://www.python.org">https://www.python.org</a>.
       </span>
       <span class="pkd-expandable" data-venv="venv" data-os="linux">
         Install python3 and python3-pip using the package manager of the Linux Distribution.
       </span>
       <span class="pkd-expandable" data-venv="none" data-os="windows">
         Install Python 3 (64-bit) from <a href="https://www.python.org/">https://www.python.org</a>.
       </span>
       <span class="pkd-expandable" data-venv="none" data-os="mac">
         Install Python 3 using <a href="https://brew.sh/">homebrew</a>
         (<code class="literal">brew install python</code>) or by manually installing the package from
         <a href="https://www.python.org">https://www.python.org</a>.
       </span>
       <span class="pkd-expandable" data-venv="none" data-os="linux">
         Install python3 and python3-pip using the package manager of the Linux Distribution.
       </span>
     </div>

Then run:

.. raw:: html

     <div class="highlight"><pre
       ><span class="pkd-expandable" data-venv="conda">conda create -n pkd python=3.8</span
       ><span class="pkd-expandable" data-venv="conda">conda activate pkd</span
       ><span class="pkd-expandable" data-venv="conda">pip install -U peekingduck</span
       ><span class="pkd-expandable" data-venv="venv" data-os="linux">python3 -m venv pkd</span
       ><span class="pkd-expandable" data-venv="venv" data-os="mac">python3 -m venv pkd</span
       ><span class="pkd-expandable" data-venv="venv" data-os="windows">python -m venv pkd</span
       ><span class="pkd-expandable" data-venv="venv" data-os="linux">source pkd/bin/activate</span
       ><span class="pkd-expandable" data-venv="venv" data-os="mac">source pkd/bin/activate</span
       ><span class="pkd-expandable" data-venv="venv" data-os="windows">pkd\Scripts\activate</span
       ><span class="pkd-expandable" data-venv="venv">pip install -U peekingduck</span
       ><span class="pkd-expandable" data-venv="none">pip install -U peekingduck</span
     ></pre></div>
   </div>

PeekingDuck supports Python 3.6 to 3.9.

It is recommended to install PeekingDuck in a Python virtual environment (such as
``pkd`` in the above commands), as it creates an isolated environment for a Python
project to install its own dependencies and avoid package version conflicts with other
projects.

.. note::

   For Apple Silicon Mac users, please see :ref:`Custom Install - Apple Silicon Mac
   <apple_silicon_mac_installation>`.

.. _verify_installation:

Verify PeekingDuck Installation
===============================

To check that PeekingDuck is installed successfully, run the following command:

.. admonition:: Terminal Session

    | \ :blue:`[~user]` \ > \ :green:`peekingduck -\-verify_install` \

You should see a video of a person waving his hand (`taken from here <https://www.youtube.com/watch?v=IKj_z2hgYUM>`_)
with bounding boxes overlaid as shown below:

.. image:: /assets/getting_started/verify_install.gif
   :class: no-scaled-link
   

| The video will auto-close when it is run to the end (about 20 seconds, depending on system speed).
| To exit earlier, click to select the video window and press :greenbox:`q`.
