{{ objname | escape | underline }}

.. rubric:: Description

.. automodule:: {{ fullname }}

{% block modules -%}
{% if modules -%}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :template: module.rst
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif -%}
{% endblock -%}

{% block classes -%}
{% if classes -%}
.. rubric:: Classes

.. autosummary::
   :toctree:
   :template: class.rst
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif -%}
{% endblock -%}

{% block functions -%}
{% if functions -%}
.. rubric:: Functions

.. autosummary::
   :toctree:
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif -%}
{% endblock -%}

{% block exceptions -%}
{% if exceptions -%}
.. rubric:: Exceptions

.. autosummary::
   :toctree:
   :template: class.rst
{% for item in exceptions %}
   {{ item }}
{%- endfor %}
{% endif -%}
{% endblock -%}
