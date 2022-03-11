{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   {% block methods %}

   {%- set meth = [] -%}
   {%- set private = [] -%}
   {%- set protected = [] -%}
   {%- set special = [] -%}

   {%- for item in methods -%}
      {%- if item != '__init__' -%}
        {{ meth.append(item) or "" }}
      {%- endif -%}
   {%- endfor -%}

   {%- for item in members -%}
      {%- if item not in inherited_members and
            item not in attributes and
            item not in meth and
            item not in ['__annotations__',
                         '__dict__',
                         '__doc__',
                         '__entries',
                         '__hash__',
                         '__init__',
                         '__members__',
                         '__module__',
                         '__slots__',
                         '__weakref__'] -%}
         {%- if item.startswith('__') and item.endswith('__') -%}
           {{ special.append(item) or "" }}
         {%- elif item.startswith('__') -%}
           {{ private.append(item) or "" }}
         {%- elif item.startswith('_') -%}
           {{ protected.append(item) or "" }}
         {%- else -%}
           {{ meth.append(item) or "" }}
         {%- endif -%}
      {%- endif -%}
   {%- endfor %}

   {%- if attributes %}
   .. rubric:: {{ _('Attributes') }}
   .. autosummary::
      :toctree:
   {% for item in attributes %}
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

   {%- if protected %}
   .. rubric:: {{ _('Protected Methods') }}
   .. autosummary::
      :toctree:
   {% for item in protected %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif -%}

   {%- if private %}
   .. rubric:: {{ _('Private Methods') }}
   .. autosummary::
      :toctree:
   {% for item in private %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif -%}

   {%- if special %}

   .. rubric:: {{ _('Special Methods') }}
   .. autosummary::
      :toctree:
   {% for item in special %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {%- endif -%}

   {%- endblock -%}
