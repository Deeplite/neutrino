.. _torch_samples:

********
Examples
********

Make sure you have installed :ref:`neutrino-engine <pypi_install_engine>`, :ref:`neutrino-torch <pypi_install_engine>`
and :ref:`deeplite-torch-zoo <pypi_install_engine>` packages successfully. Simply create ``hello_neutrino.py`` file and copy the following
sample code. This sample uses :ref:`deeplite_torch_zoo <nt_zoo>` package to make it easier for you to run Neutrino on
different scenarios, but nothing prevents you from providing your own sample models and datasets as well.

.. note::

    To download the sample codes, refer to our `neutrino-example <https://github.com/Deeplite/neutrino-examples>`_ repository.

- :ref:`classification_example`
- :ref:`od_example`
    - :ref:`od_ssd_example`
    - :ref:`od_yolo_example`
- :ref:`segmentation_example`
    - :ref:`segmentation_unet_example`
- :ref:`run_neutrino`
    - :ref:`run_one_gpu`
    - :ref:`run_multi_gpu`
    - :ref:`run_multi_multi_gpu`

.. _classification_example:

Classification Example
======================

.. literalinclude:: ../neutrino-examples/src/hello_neutrino_classifier.py
    :language: python

.. _od_example:

Object Detection Example
========================

Before you start, make sure that you were able to run the :ref:`classification_example` without any problems, as
object detection optimization is more intricate. In the following example, you will see implementations
of some interfaces required to make a non-classification task compatible with Neutrino, as explained
in :ref:`deeper`.

.. _od_ssd_example:

SSD Family
----------

.. container:: source-code-box

    .. literalinclude:: ../neutrino-examples/src/hello_neutrino_ssd.py
        :language: python
|

.. _od_yolo_example:

YOLO Family
-----------

.. container:: source-code-box

    .. literalinclude:: ../neutrino-examples/src/hello_neutrino_yolo.py
        :language: python
|

.. _segmentation_example:

Segmentation Example
====================

.. _segmentation_unet_example:

UNet family
-----------

.. container:: source-code-box

    .. literalinclude:: ../neutrino-examples/src/hello_neutrino_unet.py
        :language: python
|

.. _run_neutrino:

Run Neutrino
============

In this section we explain how you can run the engine with classification example.

.. _run_one_gpu:

Running on a single GPU
-----------------------

You can use different datasets (such as ImageNet, CIFAR100, Visual Wake Words (VWW), subset of ImageNet, MNIST) and models (such as vgg, resnet
mobilenet, etc.) from Neutrino zoo. Please see :ref:`nt_zoo` to see the list of pre-trained models and datasets.
It is recommended to first run the sample on CIFAR100 with the default values to make sure the engine works on your servers.

To run the sample:

.. code-block:: console

    $ python hello_neutrino_classifier.py --dataset cifar100 --workers 1 -a vgg19 --delta 1 --level 2 --deepsearch --batch_size 256

The output:

.. code-block:: console

    Downloading https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz to /WORKING_DIR/.neutrino-torch-zoo/cifar-100-python.tar.gz
    Extracting /WORKING_DIR/.neutrino-torch-zoo/cifar-100-python.tar.gz to /WORKING_DIR/.neutrino-torch-zoo
    2020-12-09 15:35:10 - INFO: Verifying license...
    2020-12-09 15:35:11 - INFO: The license is valid!
    Files already downloaded and verified
    Files already downloaded and verified
    2020-12-09 15:35:14 - INFO: Starting job with ID: 67CA3456
    2020-06-26 16:33:49 - INFO: Args: --dataset, cifar100, --workers, 1, -a, vgg19, --delta, 1, --level, 2, --deepsearch, --batch_size, 256
    2020-06-26 16:33:49 - INFO:
    +------------------------------------------------------------------------------------+
    | Neutrino 1.0.0                                                 26/06/2020 16:33:49 |
    +------------------------------------------------------------------------------------+
    2020-06-26 16:33:50 - INFO: Backend: TorchBackend
    2020-06-26 16:33:50 - INFO: Parsed task type 'classification'
    2020-06-26 16:33:52 - INFO: Trying forward passes on training data...
    2020-06-26 16:33:52 - INFO: ...Success
    2020-06-26 16:33:52 - INFO: Test dataset size: 10240 instances
    2020-06-26 16:33:52 - INFO: Train dataset size: 50176 instances
    2020-06-26 16:33:52 - INFO: Exporting to ONNX
    2020-12-09 15:35:17 - INFO: Model has been exported to pytorch jit format: /WORKING_DIR/ref_model_jit.pt
    2020-12-09 15:35:18 - INFO: Model has been exported to onnx format: /WORKING_DIR/ref_model.onnx
    2020-06-26 16:33:53 - INFO: Computing network status...
    2020-06-26 16:33:54 - INFO:
    +---------------------------------------------------------------+
    |                    Neutrino Model Profiler                    |
    +-----------------------------------------+---------------------+
    |            Param Name (Reference Model) |                Value|
    |                   Backend: TorchBackend |                     |
    +-----------------------------------------+---------------------+
    |          Evaluation Metric (accuracy %) |              72.4902|
    |                         Model Size (MB) |              76.6246|
    |     Computational Complexity (GigaMACs) |               0.3995|
    |         Number of Parameters (Millions) |              20.0867|
    |                   Memory Footprint (MB) |              80.2270|
    |                     Execution Time (ms) |               1.8288|
    +-----------------------------------------+---------------------+
    Note:
    * Evaluation Metric: Computed performance of the model on the given data
    * Model Size: Memory consumed by the parameters (weights and biases) of the model
    * Computational Complexity: Summation of Multiply-Add Cumulations (MACs) per single image (batch_size=1)
    * Number of Parameters: Total number of parameters (trainable and non-trainable) in the model
    * Memory Footprint: Total memory consumed by the parameters (weights and biases) and activations (per layer) per single image (batch_size=1)
    * Execution Time: On current device, time required for the forward pass per single image (batch_size=1)
    +---------------------------------------------------------------+
    2020-06-26 16:33:54 - INFO: Analyzing design space...
    2020-06-26 16:33:56 - INFO:
    +------------------------------------------------------------------------------------+
    |                                  Target |                           71.49 accuracy |
    +------------------------------------------------------------------------------------+
    |                           At most steps |                                        7 |
    +------------------------------------------------------------------------------------+
    |              Estimated exploration time |                    2:31:27 (d, hh:mm:ss) |
    +------------------------------------------------------------------------------------+
    2020-06-26 16:33:56 - INFO: Phase 1
    2020-06-26 16:33:56 - INFO: Step 1
    2020-06-26 16:37:25 - INFO: Starting ... [0%]
    2020-06-26 16:41:48 - INFO: Exploring .. [25%]
    2020-06-26 16:46:15 - INFO: Exploring .. [50%]
    2020-06-26 16:50:44 - INFO: Exploring .. [75%]
    2020-06-26 16:55:14 - INFO: Done ... [100%]
    2020-06-26 16:55:15 - INFO: Step 2
    2020-06-26 17:00:06 - INFO: Starting ... [0%]
    2020-06-26 17:04:17 - INFO: Exploring .. [25%]
    2020-06-26 17:08:28 - INFO: Exploring .. [50%]
    2020-06-26 17:12:40 - INFO: Exploring .. [75%]
    2020-06-26 17:16:51 - INFO: Done ... [100%]
    2020-06-26 17:16:52 - INFO: Step 3
    2020-06-26 17:21:09 - INFO: Starting ... [0%]
    2020-06-26 17:25:31 - INFO: Exploring .. [25%]
    2020-06-26 17:29:53 - INFO: Exploring .. [50%]
    2020-06-26 17:34:16 - INFO: Exploring .. [75%]
    2020-06-26 17:38:38 - INFO: Done ... [100%]
    2020-06-26 17:38:40 - INFO: Step 4
    2020-06-26 17:42:37 - INFO: Starting ... [0%]
    2020-06-26 17:47:05 - INFO: Exploring .. [25%]
    2020-06-26 17:51:32 - INFO: Exploring .. [50%]
    2020-06-26 17:55:59 - INFO: Exploring .. [75%]
    2020-06-26 18:00:25 - INFO: Done ... [100%]
    2020-06-26 18:00:26 - INFO: Phase 2
    2020-06-26 18:00:26 - INFO: Step 1
    2020-06-26 18:00:26 - INFO: Starting ... [0%]
    2020-06-26 18:04:38 - INFO: Exploring .. [25%]
    2020-06-26 18:08:50 - INFO: Exploring .. [50%]
    2020-06-26 18:13:02 - INFO: Exploring .. [75%]
    2020-06-26 18:17:14 - INFO: Done ... [100%]
    2020-06-26 18:17:16 - INFO: Step 2
    2020-06-26 18:17:16 - INFO: Starting ... [0%]
    2020-06-26 18:21:40 - INFO: Exploring .. [25%]
    2020-06-26 18:26:04 - INFO: Exploring .. [50%]
    2020-06-26 18:30:28 - INFO: Exploring .. [75%]
    2020-06-26 18:33:16 - INFO: Done ... [100%]
    2020-06-26 18:33:17 - INFO: Step 3
    2020-06-26 18:33:17 - INFO: Starting ... [0%]
    2020-06-26 18:37:46 - INFO: Exploring .. [25%]
    2020-06-26 18:42:14 - INFO: Exploring .. [50%]
    2020-06-26 18:46:43 - INFO: Exploring .. [75%]
    2020-06-26 18:46:43 - INFO: Done ... [100%]
    2020-06-26 18:46:44 - INFO: Step 4
    2020-06-26 18:46:44 - INFO: Starting ... [0%]
    2020-06-26 18:51:10 - INFO: Exploring .. [25%]
    2020-06-26 18:55:36 - INFO: Exploring .. [50%]
    2020-06-26 19:00:01 - INFO: Exploring .. [75%]
    2020-06-26 19:04:16 - INFO: Done ... [100%]
    2020-06-26 19:04:18 - INFO: Comparing networks status...
    2020-06-26 19:04:20 - INFO:
    +--------------------------------------------------------------------------------------------------------------------------+
    |                                                 Neutrino Model Profiler                                                  |
    +-----------------------------------------+--------------------------+--------------------------+--------------------------+
    |                              Param Name |               Enhancement|   Value (Optimized Model)|   Value (Reference Model)|
    |                                         |                          |     Backend: TorchBackend|     Backend: TorchBackend|
    +-----------------------------------------+--------------------------+--------------------------+--------------------------+
    |          Evaluation Metric (accuracy %) |                   -0.6543|                   71.8359|                   72.4902|
    |                         Model Size (MB) |                    28.14x|                    2.7229|                   76.6246|
    |     Computational Complexity (GigaMACs) |                     5.34x|                    0.0748|                    0.3995|
    |         Number of Parameters (Millions) |                    28.14x|                    0.7138|                   20.0867|
    |                   Memory Footprint (MB) |                     1.98x|                    3.8986|                    7.7362|
    |                     Execution Time (ms) |                     1.60x|                    0.0376|                    0.0603|
    +-----------------------------------------+--------------------------+--------------------------+--------------------------+
    Note:
    * Evaluation Metric: Computed performance of the model on the given data
    * Model Size: Memory consumed by the parameters (weights and biases) of the model
    * Computational Complexity: Summation of Multiply-Add Cumulations (MACs) per single image (batch_size=1)
    * Number of Parameters: Total number of parameters (trainable and non-trainable) in the model
    * Memory Footprint: Total memory consumed by parameters and activations per single image (batch_size=1)
    * Execution Time: On current device, time required for the forward pass per single image
    +--------------------------------------------------------------------------------------------------------------------------+
    2020-12-09 15:46:59 - INFO: The engine successfully optimized your reference model, enjoy!
    2020-12-09 15:46:59 - INFO: Exporting to Native and ONNX formats
    2020-12-09 15:46:59 - INFO: Model has been exported to pytorch jit format: /WORKING_DIR/opt_model_jit.pt
    2020-12-09 15:47:00 - INFO: Model has been exported to onnx format: /WORKING_DIR/opt_model.onnx
    2020-12-09 15:47:00 - INFO: Job with ID 67CA3456 finished
    2020-06-26 19:04:21 - INFO: Total execution time: 2:30:32 (d, hh:mm:ss)
    2020-12-09 15:47:00 - INFO: Log has been exported to: $NEUTRINO_HOME/logs/67CA3456-2020-12-09.log

.. figure:: media/demo_run.gif
   :align: center

.. note::

    Please note that the exploration problem is a hard problem which makes it almost impossible to estimate a precise
    exploration time. The engine reports an estimate for the total exploration time as "Estimated exploration time". Users often find the actual time is shorter than the estimated time.
    So, it is normal if "Estimated exploration time" and "Total execution time" are different.

.. _run_multi_gpu:

Running on multi-gpu on a single machine
----------------------------------------

.. important::

    Currently, the multi-GPU support is available only for the Production version of Deeplite Neutrino. Refer, :ref:`how to upgrade <feature_comparison>`.

Neutrino leverages `Horovod <https://github.com/horovod/horovod>`_ for distributed training. Horovod is a distributed
deep learning training framework for TensorFlow, Keras, PyTorch, and Apache MXNet. The goal of Horovod is to make distributed
deep learning fast and easy to use. To enable distributed training in Neutrino engine you need to make sure Horovod and its dependencies are installed correctly on your
servers. We have prepared Dockerfile on top of Horovod docker so you can get started with Neutrino and Horovod in mintues.

Start the optimization process and specify the number of workers on the command line as you normally would when using
Horovod (for more information please visit `Horovod in Docker <https://github.com/horovod/horovod/blob/master/docs/docker.rst#running-on-a-single-machine>`_).

* Get the docker file from `here <https://github.com/Deeplite/neutrino/blob/master/Dockerfile.gpu>`_.
* Build your docker image:

.. code-block:: console
    
    sudo docker build -t neutrino:latest -f Dockerfile.gpu .

* Run the image with `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ command:

.. code-block:: console

    sudo nvidia-docker run -it --shm-size=32g  -v /home/JohnDoe/:/neutrino  neutrino:latest

* To test Neutrino™ in distributed mode, from within your docker container, checkout and navigate to the `neutrino-examples <https://github.com/Deeplite/neutrino-examples>`_ repository, then run the following command:

.. code-block:: console

    horovodrun -np 1 -H localhost:1 python src/hello_neutrino_classifier.py --arch resnet18 --dataset cifar100 --delta 1 --horovod

* Another example with 4 GPUs:

.. code-block:: console

    $ horovodrun -np 4 -H localhost:4 python hello_neutrino.py --dataset cifar100 --workers 1 -a vgg19 --delta 1 --level 2 --deepsearch --horovod --batch_size 256

.. _run_multi_multi_gpu:

Running on multi-gpu on multiple machines
-----------------------------------------

.. important::

    Currently, the multi-GPU support on multiple machines is available only for the Production version of Deeplite Neutrino. Refer, :ref:`how to upgrade <feature_comparison>`.

.. code-block:: console

    $ horovodrun -np 8 -H hostname1:4,hostname2:4 python hello_neutrino.py --dataset cifar100 --workers 1 -a vgg19 --delta 1 --level 2 --deepsearch --horovod --batch_size 256

`Horovod on multiple machines <https://github.com/horovod/horovod/blob/master/docs/docker.rst#running-on-multiple-machines>`_

* Make sure to set **--horovod** in the config.
* By default Neutrino uses fp16 compression setting for Horovod for inter core communication. To use fp32 please set the ``$HVD_FP16`` environment variable to 0.

.. warning::

    The performance of your network might be impacted by distributed training if you don’t use an appropriate batch size.

