*******
Privacy Protection (People & Computer Screens)
*******

Overview
========

Videos and Pictures often contain people and other sensitive visual information (e.g., the display
on computer screens), even though this information might not be needed at all for visual
processing. Our solution performs full body anonymization and computer screen blurring to protect
the identities of individuals, and the sensitive information on computer screens from footage or
pictures. It can be used to comply with the General Data Protection Regulation (GDPR) or other data
privacy laws.

..
    Use case demo gif

.. image:: /assets/use_cases/demo_use_case.gif
   :class: no-scaled-link
   :width: 100 %

Our solution automatically detects people, laptop and computer screens, and blurs them. This is
explained in the `How It Works`_ section.

Demo
====

..
    Replace <use_case_config> with the actual name

.. |pipeline_config| replace:: <use_case_config>.yml
.. _pipeline_config: https://github.com/aimakerspace/PeekingDuck/blob/main/use_cases/<use_case_config>.yml

To try our solution on your own computer, :doc:`install </getting_started/02_standard_install>` and run
PeekingDuck with the configuration file |pipeline_config|_ as shown:

.. admonition:: Terminal Session

    | \ :blue:`[~user]` \ > \ :green:`peekingduck run -\-config_path <path/to/`\ |pipeline_config|\ :green:`>`

How It Works
============

There are <number> main components to our solution:

#. Human and Computer Screen Segmentation, and
#. Human and Computer Screen blurring.

**1. Human and Computer Screen Segmentation**

We use an open source instance segmentation model known as <to-be-inserted-with-link> to obtain the
masks of persons, computer screens and laptops. The masks are akin to the input frames or images,
except that it only has a single channel and each pixel on the mask is a binary of either 1 or 0,
which indicates whether a specific class of thing is present (1) or absent (0) in a particular
location of the image. For more information on how to adjust the <insert-model-name> node, check
out the <insert-link-to-model-page>.

**2. Human and Computer Screen blurring**

To blur the people and computer screens, we pixelate or gaussian blur the image pixels where the
pixel values of the relevant masks are equal to 1 (Presence of object).

Nodes Used
==========

These are the nodes used in the earlier demo (also in |pipeline_config|_):

.. code-block:: yaml

   nodes:
   - node.one
   - node.two
   - node.three
   
**1. Instance Segmentation Node**

<content>

**#. Adjusting Nodes**

Some common node behaviors that you might want to adjust are:

* ``param_1``: Definition
* ``param_2``: Definition

