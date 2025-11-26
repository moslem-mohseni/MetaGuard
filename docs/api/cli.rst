CLI Reference
=============

**Author:** Moslem Mohseni

MetaGuard provides a powerful command-line interface for fraud detection.

Installation
------------

Install with CLI support:

.. code-block:: bash

    pip install metaguard[cli]

Global Options
--------------

.. code-block:: bash

    metaguard --help        # Show help
    metaguard --version     # Show version
    metaguard -v            # Show version (short)

Commands
--------

detect
~~~~~~

Detect fraud in a single transaction.

.. code-block:: bash

    metaguard detect --amount 5000 --hour 3 --user-age 5 --tx-count 50

Options:

.. code-block:: text

    -a, --amount      Transaction amount (required)
    -h, --hour        Hour of day 0-23 (required)
    -u, --user-age    Account age in days (required)
    -t, --tx-count    Number of prior transactions (required)
    -j, --json        Output as JSON

Example output:

.. code-block:: text

    ╭─────────────── Detection Result ───────────────╮
    │ Status: SUSPICIOUS                             │
    │ Risk Score: 99.00%                             │
    │ Risk Level: High                               │
    ╰────────────────────────────────────────────────╯

JSON output:

.. code-block:: bash

    metaguard detect -a 5000 -h 3 -u 5 -t 50 --json

.. code-block:: json

    {
        "is_suspicious": true,
        "risk_score": 0.99,
        "risk_level": "High"
    }

analyze
~~~~~~~

Perform detailed risk analysis.

.. code-block:: bash

    metaguard analyze --amount 5000 --hour 3 --user-age 5 --tx-count 50

Example output:

.. code-block:: text

    Risk Score: 100.0/100
    Risk Level: High
    Active Factors: 4/4

           Risk Factors
    ┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
    ┃ Factor          ┃ Status   ┃
    ┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
    │ High Amount     │ Active   │
    │ New Account     │ Active   │
    │ High Frequency  │ Active   │
    │ Unusual Hour    │ Active   │
    └─────────────────┴──────────┘

batch
~~~~~

Process multiple transactions from a JSON file.

.. code-block:: bash

    metaguard batch transactions.json

With output file:

.. code-block:: bash

    metaguard batch transactions.json --output results.json

Input format (``transactions.json``):

.. code-block:: json

    [
        {"amount": 100, "hour": 10, "user_age_days": 100, "transaction_count": 5},
        {"amount": 5000, "hour": 3, "user_age_days": 5, "transaction_count": 50}
    ]

Or with wrapper:

.. code-block:: json

    {
        "transactions": [
            {"amount": 100, "hour": 10, "user_age_days": 100, "transaction_count": 5}
        ]
    }

Output:

.. code-block:: text

                Batch Processing Results
    ┏━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┓
    ┃ #  ┃ Amount     ┃ Status     ┃ Risk Level ┃ Score  ┃
    ┡━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━┩
    │ 1  │ $100.00    │ OK         │ Low        │ 15.00% │
    │ 2  │ $5,000.00  │ SUSPICIOUS │ High       │ 99.00% │
    └────┴────────────┴────────────┴────────────┴────────┘

    Summary: 1/2 suspicious (50.0%)

info
~~~~

Show model and configuration information.

.. code-block:: bash

    metaguard info

Output:

.. code-block:: text

                 MetaGuard Information
    ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ Property       ┃ Value                          ┃
    ┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │ Version        │ 1.1.0                          │
    │ Model Type     │ RandomForestClassifier         │
    │ Model Path     │ /path/to/model.pkl             │
    │ Risk Threshold │ 50.0%                          │
    └────────────────┴────────────────────────────────┘

serve
~~~~~

Start the REST API server.

.. code-block:: bash

    # Default
    metaguard serve

    # Custom host and port
    metaguard serve --host 0.0.0.0 --port 8080

    # Development mode with hot reload
    metaguard serve --reload

Options:

.. code-block:: text

    --host         Host to bind to (default: 0.0.0.0)
    -p, --port     Port to bind to (default: 8000)
    -r, --reload   Enable auto-reload for development

Exit Codes
----------

.. list-table::
   :header-rows: 1

   * - Code
     - Description
   * - 0
     - Success
   * - 1
     - Error (invalid input, file not found, etc.)

Shell Completion
----------------

Bash
~~~~

.. code-block:: bash

    # Add to ~/.bashrc
    eval "$(_METAGUARD_COMPLETE=bash_source metaguard)"

Zsh
~~~

.. code-block:: bash

    # Add to ~/.zshrc
    eval "$(_METAGUARD_COMPLETE=zsh_source metaguard)"

PowerShell
~~~~~~~~~~

.. code-block:: powershell

    # Add to profile
    Register-ArgumentCompleter -Native -CommandName metaguard -ScriptBlock {
        param($wordToComplete, $commandAst, $cursorPosition)
        $env:_METAGUARD_COMPLETE = "powershell_complete"
        metaguard | ForEach-Object { $_ }
    }

Integration Examples
--------------------

CI/CD Pipeline
~~~~~~~~~~~~~~

.. code-block:: yaml

    # GitHub Actions example
    - name: Check transactions
      run: |
        metaguard batch ./data/transactions.json --output results.json
        suspicious=$(jq '.suspicious' results.json)
        if [ "$suspicious" -gt 0 ]; then
          echo "Found $suspicious suspicious transactions!"
          exit 1
        fi

Cron Job
~~~~~~~~

.. code-block:: bash

    # Check daily transactions at midnight
    0 0 * * * /usr/local/bin/metaguard batch /data/daily.json -o /reports/$(date +\%Y\%m\%d).json

Python Subprocess
~~~~~~~~~~~~~~~~~

.. code-block:: python

    import subprocess
    import json

    result = subprocess.run(
        ["metaguard", "detect", "-a", "5000", "-h", "3", "-u", "5", "-t", "50", "--json"],
        capture_output=True,
        text=True
    )

    data = json.loads(result.stdout)
    if data["is_suspicious"]:
        print(f"Alert! Risk score: {data['risk_score']:.2%}")
