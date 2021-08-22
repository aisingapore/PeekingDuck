.. PeekingDuck documentation master file, created by
   sphinx-quickstart on Tue Jun  8 16:33:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PeekingDuck's documentation!
=======================================

.. toctree::
   :titlesonly:
   :caption: Introduction

   index.md
   introduction/02_bibliography.md

.. toctree::
   :titlesonly:
   :glob:
   :caption: Getting Started

   getting_started/*

.. toctree::
   :titlesonly:
   :glob:
   :caption: Model Resources & Information

   resources/*

.. toctree::
   :titlesonly:
   :glob:
   :caption: Use Cases

   use_cases/*

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
   peekingduck.pipeline.nodes.model
   peekingduck.pipeline.nodes.dabble
   peekingduck.pipeline.nodes.draw
   peekingduck.pipeline.nodes.output
