*************
Basic Install
*************

.. |br| raw:: html

   <br />

This section covers the basic PeekingDuck installation that is suitable for most users.

Install PeekingDuck
===================

    Install PeekingDuck from `PyPI <https://pypi.org/project/peekingduck>`_ 
    for Windows (10 / 11), Linux and MacOS (Intel)::

    > pip install peekingduck

    .. note::
        For M1 Mac users, please see :ref:`Advanced Install - M1 Mac <m1_mac_installation>`


.. _verify_installation:

Verify PeekingDuck Installation
===============================

    To check that PeekingDuck is installed successfully, create a project folder at
    a convenient location::

    > mkdir pkd_project
    > cd pkd_project

    Then, initialize a PeekingDuck project and run it::

    > peekingduck init
    > peekingduck run

    You should see a video of a person waving hand (`taken from here
    <https://www.youtube.com/watch?v=IKj_z2hgYUM>`_).

    The video will auto-close when it is runs to the end (about 18 seconds). |br|
    To exit earlier, click to select the video window and press ``q``.

