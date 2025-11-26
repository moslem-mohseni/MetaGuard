Installation
============

Requirements
------------

MetaGuard requires Python 3.9 or later.

Dependencies:

- numpy
- pandas
- scikit-learn
- pydantic

Installing from PyPI
--------------------

The simplest way to install MetaGuard is from PyPI:

.. code-block:: bash

   pip install metaguard

Installing with Optional Dependencies
-------------------------------------

For development (includes testing and linting tools):

.. code-block:: bash

   pip install metaguard[dev]

For documentation building:

.. code-block:: bash

   pip install metaguard[docs]

For API server (FastAPI):

.. code-block:: bash

   pip install metaguard[api]

For XGBoost model support:

.. code-block:: bash

   pip install metaguard[xgboost]

Install everything:

.. code-block:: bash

   pip install metaguard[dev,docs,api,xgboost]

Installing from Source
----------------------

Clone the repository:

.. code-block:: bash

   git clone https://github.com/moslem-mohseni/MetaGuard.git
   cd MetaGuard

Install in development mode:

.. code-block:: bash

   pip install -e ".[dev]"

Verifying Installation
----------------------

After installation, verify it works:

.. code-block:: python

   >>> import metaguard
   >>> print(metaguard.__version__)
   1.1.0
   >>> from metaguard import check_transaction
   >>> result = check_transaction({
   ...     "amount": 100,
   ...     "hour": 14,
   ...     "user_age_days": 30,
   ...     "transaction_count": 5
   ... })
   >>> print(result["is_suspicious"])
   False

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**ImportError: No module named 'sklearn'**

Install scikit-learn:

.. code-block:: bash

   pip install scikit-learn

**Model file not found**

MetaGuard comes with a pre-trained model. If you see this error, try reinstalling:

.. code-block:: bash

   pip uninstall metaguard
   pip install metaguard

**Python version error**

MetaGuard requires Python 3.9+. Check your version:

.. code-block:: bash

   python --version

Upgrading
---------

To upgrade to the latest version:

.. code-block:: bash

   pip install --upgrade metaguard
