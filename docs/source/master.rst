.. PeekingDuck documentation master file, created by
   sphinx-quickstart on Tue Jun  8 16:33:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PeekingDuck's documentation!
=======================================

.. toctree::
   :caption: Contents

   index.md
   getting_started/index
   resources/index
   use_cases/index

Nodes
=====================================
Nodes are the core of PeekingDuck. See below for the readily-available nodes and their references.

.. autosummary::
   :toctree:
   :caption: API Documentation
   :template: module.rst
   :recursive:

   peekingduck
   peekingduck.pipeline.nodes.input
   peekingduck.pipeline.nodes.preprocess
   peekingduck.pipeline.nodes.model
   peekingduck.pipeline.nodes.dabble
   peekingduck.pipeline.nodes.draw
   peekingduck.pipeline.nodes.output
