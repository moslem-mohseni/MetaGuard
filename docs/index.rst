MetaGuard Documentation
=======================

.. image:: https://badge.fury.io/py/metaguard.svg
   :target: https://badge.fury.io/py/metaguard
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/metaguard.svg
   :target: https://pypi.org/project/metaguard/
   :alt: Python Version

.. image:: https://img.shields.io/github/license/moslem-mohseni/MetaGuard.svg
   :target: https://github.com/moslem-mohseni/MetaGuard/blob/main/LICENSE
   :alt: License

**MetaGuard** is a fraud detection library for metaverse transactions. Detect suspicious
activity with just 3 lines of Python code.

Quick Start
-----------

.. code-block:: python

   from metaguard import check_transaction

   result = check_transaction({
       "amount": 5000,
       "hour": 3,
       "user_age_days": 5,
       "transaction_count": 50
   })

   print(f"Suspicious: {result['is_suspicious']}")
   print(f"Risk Level: {result['risk_level']}")

Features
--------

- **Simple API**: Detect fraud with just 3 lines of code
- **ML-Powered**: Pre-trained Random Forest model
- **Batch Processing**: Process multiple transactions efficiently
- **Risk Analysis**: Get detailed risk factors and scores
- **Configurable**: Customize thresholds and model parameters
- **Type-Safe**: Full type hints for IDE support
- **Well-Tested**: >90% test coverage

Installation
------------

.. code-block:: bash

   pip install metaguard

For development:

.. code-block:: bash

   pip install metaguard[dev]

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/contributing
   development/testing
   architecture
   changelog

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

About
-----

MetaGuard is developed and maintained by `Moslem Mohseni <https://github.com/moslem-mohseni>`_.

License
-------

MetaGuard is released under the MIT License. See the LICENSE file for details.
