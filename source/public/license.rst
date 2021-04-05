***********************************
Get Community or Production License
***********************************

* :ref:`community_license`
* :ref:`production_license`
* :ref:`use_license`
    * :ref:`use_license_online`
    * :ref:`use_license_offline`

.. _community_license:

Get a Free Community License
============================

The community license key is completely free-to-obtain and free-to-use. `Fill out this simple form <https://info.deeplite.ai/community>`_ to obtain the license key for the Community Version of Deeplite Neutrino™.

.. _production_license:

Get a Production License
========================

.. important::

    Obtaining the license key is necessary for using the production version of Neutrino and use some advanced feature capabilities.

To obtain the license key for the Production version of Deeplite Neutrino™, kind reach out to us `via email <support@deeplite.ai>`_

.. _use_license:

How to Use the License
======================

.. _use_license_online:

Online Key
----------

You need a valid license to use Neutrino engine. After you get your license key (license key format: AAAAA-AAAAA-AAAAA-AAAAA)
please set the ``$NEUTRINO_LICENSE`` env var to that value.

.. code-block:: console

    $ export NEUTRINO_LICENSE=LICENSE_KEY

.. _use_license_offline:

Offline Key
-----------

If you have no or limited internet availability, you can obtain a license file so that Neutrino can verify the license information offline. To do so:

- Go to the `Neutrino activation form <https://app.cryptolens.io/Form/A/upv5bDrZ/1062>`_.
- Enter your license key.
- Obtain your machine code then enter it in the activation form. You can get your machine code by opening a python terminal where neutrino-engine has been installed and running the the following commands:

.. code-block:: python

    from licensing.methods import Helpers
    Helpers.GetMachineCode()

- Click on activate. The license file will be downloaded.
- Copy the license file to location of your choice and then set the ``$NEUTRINO_LICENSE_FILE`` env variable to the absolute path of this location.

.. code-block:: console

    $ export $NEUTRINO_LICENSE_FILE=LICENSE_FILE_PATH

To check your license value:

.. code-block:: python

    import neutrino.engine
    neutrino.engine.__license__


.. note::

    You can also set ``$NEUTRINO_LICENSE`` or ``$NEUTRINO_LICENSE_FILE`` in your ``./bashrc`` file as well `(How do I set environment variables?) <https://askubuntu.com/questions/730/how-do-i-set-environment-variables>`_.
