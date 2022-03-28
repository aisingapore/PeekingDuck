.. include:: /include/data_type.rst
.. include:: /include/substitution.rst

{{ fullname | escape | underline }}

.. rubric:: Description

.. automodule:: {{ fullname }}

{% if not fullname.split(".")[-1] in ["dabble", "draw", "input", "model",
                                      "output", "augment"] -%}
.. autoclass:: {{ fullname }}.Node
   :members:
   :exclude-members: run, release_resources

   {% block methods -%}
      {% if methods -%}
.. rubric:: Methods
         {% for item in all_methods -%}
            {%- if not item.startswith('_') %}
               ~{{ name }}.{{ item }}
            {%- endif -%}
         {%- endfor %}
      {% endif -%}
   {% endblock -%}
{% else -%}
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
{% endif -%}

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
