..
    Template file that contains some boilerplate text and substitutions for a use case write-up.

.. |title| replace:: Use Case Title

*******
|title|
*******

Overview
========

<Introduction / background / rationale>

..
    Use case demo gif

.. image:: /assets/use_cases/demo_use_case.gif
   :class: no-scaled-link
   :width: 100 %

<General one-liner or two on the solution>. This is explained in the `How it Works`_ section.

Demo
====

..
    Replace <use_case_config> with the actual name

.. |pipeline_config| replace:: <use_case_config>.yml
.. _pipeline_config: https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/<use_case_config>.yml

To try our solution on your own computer, :doc:`install </getting_started/01_installation>` and run
PeekingDuck with the configuration file |pipeline_config|_ as shown:

.. parsed-literal::

    > peekingduck run --config_path <path/to/\ |pipeline_config|\ >

How it Works
============

There are <number> main components to our solution:

#. The first component,
#. The second component, and
#. The last component.

**1. The First Component**

<content>

**2. The Second Component**

<content>

**3. The Last Component**

<content>

Nodes Used
==========

These are the nodes used in the earlier demo (also in |pipeline_config|_):

.. code-block:: yaml

   nodes:
   - node.one
   - node.two
   - node.three
   
**1. The First Node**

<content>

**#. Adjusting Nodes**

Some common node behaviors that you might want to adjust are:

* ``param_1``: Definition
* ``param_2``: Definition
