.. PeekingDuck documentation master file, created by
   sphinx-quickstart on Tue Jun  8 16:33:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/peekingduck.png
   :height: 300
   :width: 300
   :align: center
   :alt: PeekingDuck logo

Welcome to PeekingDuck's documentation!
=======================================

.. toctree::
   :titlesonly:
   :caption: Introduction

   introduction/01_introduction.md
   introduction/02_bibliography.md

.. toctree::
   :titlesonly:
   :glob:
   :caption: Getting Started

   getting_started/*

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
   :caption: API documentation
   :template: module.rst
   :recursive:

   peekingduck.pipeline.nodes.input
   peekingduck.pipeline.nodes.model
   peekingduck.pipeline.nodes.dabble
   peekingduck.pipeline.nodes.draw
   peekingduck.pipeline.nodes.output


API documentation
=======================================

.. autosummary::
   :toctree:
   :caption: API documentation
   :template: module.rst
   :recursive:

   peekingduck