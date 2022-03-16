************************
Documentation Convention
************************

.. include:: /include/substitution.rst


.. _documentation_convention:

Parts of this documentation and the tutorials are run from the command line
interface (CLI) environment, e.g., via `Terminal` in Linux/macOS, or via
`Anaconda` in Windows.
There will be examples of commands you need to type as inputs and text that
PeekingDuck will display as outputs. The input commands can be dependent on the 
current folder where they are typed.

The following text color scheme is used to illustrate these different contexts:

+----------------+------------------------------+-----------------------------------+
| Color          | Context                      | Example                           |
+================+==============================+===================================+
| :blue:`Blue`   | Current folder               | :blue:`[~user/src]`               |
+----------------+------------------------------+-----------------------------------+
| :green:`Green` | User input: what you type in | > :green:`peekingduck -\-version` |
+----------------+------------------------------+-----------------------------------+
| Black          | PeekingDuck's output         | peekingduck, version v1.2.0       |
+----------------+------------------------------+-----------------------------------+

The command prompt is assumed to be the symbol ``>``,
your home directory is assumed to be ``~user``,
and the symbol \ :green:`⏎` \ means to press the ``<Enter>`` key.

Putting it altogether, a sample terminal session looks like this:

.. admonition:: Terminal Session

    | \ :blue:`[~user/src]` \ > \ :green:`peekingduck -\-version` \
    | peekingduck, version v1.2.0

