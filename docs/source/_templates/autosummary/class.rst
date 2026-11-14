{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   {% block methods %}

   {%- set meth = [] -%}
   {%- set inherited_meth = [] -%}
   {%- for item in methods if not item.startswith('_') -%}
      {%- if item in inherited_members -%}
         {{ inherited_meth.append(item) or "" }}
      {%- else -%}
         {{ meth.append(item) or "" }}
      {%- endif -%}
   {%- endfor %}

   {%- if attributes %}
   .. rubric:: {{ _('Attributes') }}
   .. autosummary::
      :toctree:
   {% for item in attributes if not item.startswith('_') %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif -%}

   {%- if meth %}
   .. rubric:: {{ _('Public Methods') }}
   .. autosummary::
      :toctree:
   {% for item in meth %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif -%}

   {%- if inherited_meth %}

   .. rubric:: {{ _('Inherited Methods') }}
   .. autosummary::
      :toctree:
   {% for item in inherited_meth %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {%- endif -%}

   {%- endblock -%}
