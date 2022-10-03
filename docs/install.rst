************
Installation
************

- :ref:`pypi_install`
    - :ref:`pypi_install_engine`
    - :ref:`pypi_install_torch`
    - :ref:`pypi_install_zoo`
    - :ref:`pypi_update`
    - :ref:`pypi_uninstall`

.. _pypi_install:

Through PyPi Repository
=======================

Use ``pip`` to install Neutrino engine from our internal PyPi repository. We recommend creating a new python virtualenv,
then pip install using the following commands, entering your credentials when prompted.

.. important::
    We currently only support pip installation for linux environment or Windows Subsystem for Linux (WSL).
    Windows/mac installation is currently not supported.

.. _pypi_install_engine:

Install Neutrino Engine
-----------------------

To install Neutrino engine:

.. code-block:: console

    $ pip install --upgrade pip
    $ pip install neutrino-engine

**Minimal Dependencies**

- Python 3.6+
- Ubuntu>=16.04
- neutrino-profiler==1.0.0
- scipy==1.4.1
- numpy==1.18.5
- tensorly==0.4.5
- pyyaml==5.3.1
- onnx==1.7.0
- ordered_set==4.0.2
- licensing==0.26
- grpcio==1.29.0
- google-cloud-logging==1.15.1
- pyAesCrypt==0.4.3
- cryptography==3.4.6

.. _pypi_install_torch:

Install Torch Backend
---------------------

To install `torch` backend:

.. code-block:: console

    $ pip install --upgrade pip
    $ pip install neutrino-torch

**Minimal Dependencies:**

- neutrino-engine>=5.3.2
- torch==1.4.0
- ptflops==0.6.2
- Augmentor==0.2.8"

.. _pypi_install_zoo:

(optional) Install deeplite-torch-zoo
-------------------------------------

Install ``deeplite-torch-zoo`` to get access to our entire list of models and pretrained models, as follows

.. code-block:: console

    $ pip install --upgrade pip
    $ pip install deeplite-torch-zoo

See :ref:`nt_zoo` for additional installation details.

.. _pypi_update:

How to Update
-------------

It is recommended to update your package regularly. To update the package run:

.. code-block:: console

    $ pip install --upgrade neutrino-engine 
    $ pip install --upgrade neutrino-torch
    $ pip install --upgrade deeplite-torch-zoo

.. _pypi_uninstall:

How to Uninstall
----------------

To uninstall the package run:

.. code-block:: console

    $ pip uninstall neutrino-engine
    $ pip uninstall neutrino-torch
    $ pip uninstall deeplite-torch-zoo

.. note::

    To install/update the requirements manually use ``--no-dependencies`` flag in the commands above.
    Otherwise, they will be installed/upgraded automatically.
