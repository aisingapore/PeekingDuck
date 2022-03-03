.. include:: /include/data_type.rst
.. include:: /include/substitution.rst

{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :exclude-members: run

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
