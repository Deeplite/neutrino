.. _nt_zoo:

******************
deeplite-torch-zoo
******************

The ``deeplite-torch-zoo`` package is a collection of popular CNN model architectures and benchmark datasets for PyTorch framework. The models are grouped under different datasets and different task types such as classification, object detection, and segmentation. The primary aim of this ``deeplite-torch-zoo`` is to booststrap applications by starting with the most suitable pretrained models. In addition, the pretrained models from ``deeplite-torch-zoo`` can be used as a good starting point for optimizing model architectures using our :ref:`neutrino_engine`.

* :ref:`zoo_install`
    * :ref:`zoo_install_pip`
    * :ref:`zoo_install_source`
    * :ref:`zoo_install_dev`
* :ref:`zoo_usage`
    * :ref:`zoo_usage_load_dataset`
        * :ref:`zoo_usage_load_dataset_classification`
        * :ref:`zoo_usage_load_dataset_od`
    * :ref:`zoo_usage_load_models`
        * :ref:`zoo_usage_load_models_classification`
        * :ref:`zoo_usage_load_models_od`
* :ref:`zoo_available_models`
* :ref:`zoo_available_datasets`
* :ref:`zoo_contribute`
* :ref:`zoo_benchmark_results`
    * :ref:`zoo_benchmark_results_voc_od`
    * :ref:`zoo_benchmark_results_voc_seg`
    * :ref:`zoo_benchmark_results_mnist`
    * :ref:`zoo_benchmark_results_cifar100`
    * :ref:`zoo_benchmark_results_vww`
    * :ref:`zoo_benchmark_results_imagenet10`
    * :ref:`zoo_benchmark_results_imagenet16`
    * :ref:`zoo_benchmark_results_imagenet`


.. _zoo_install:

Installation
============

.. _zoo_install_pip:

1. Install using pip
--------------------

Use following command to install the package from our internal PyPI repository.

.. code-block:: console

    $ pip install --upgrade pip
    $ pip install deeplite-torch-zoo

.. _zoo_install_source:

2. Install from source
----------------------

.. code-block:: console

    $ git clone https://github.com/Deeplite/deeplite-torch-zoo.git
    $ pip install .

.. _zoo_install_dev:

3. Install in Dev mode
----------------------

.. code-block:: console

    $ git clone https://github.com/Deeplite/deeplite-torch-zoo.git
    $ pip install -e .
    $ pip install -r requirements-test.txt

To test the installation, one can run the basic tests using `pytest` command in the root folder.

**Minimal Dependencies**

- torch>=1.4,<=1.8.1
- opencv-python
- scipy>=1.4.1
- numpy==1.19.5
- pycocotools==2.0.4
- Cython==0.29.30
- tqdm==4.46.0
- albumentations
- pretrainedmodels==0.7.4
- torchfcn==1.9.7
- tensorboardX==2.4.1
- pyvww==0.1.1
- timm==0.5.4
- texttable==1.6.4
- pytz
- torchmetrics==0.8.0
- mean_average_precision==2021.4.26.0
- ptflops==0.6.2

.. _zoo_usage:

How to Use
==========

The ``deeplite-torch-zoo`` is collection of benchmark computer vision datasets and pretrained models. The main API functions provided in the zoo are

.. code-block:: python

    from deeplite_torch_zoo import get_data_splits_by_name  # create dataloaders
    from deeplite_torch_zoo import get_model_by_name  # get a pretrained model for a task
    from deeplite_torch_zoo import get_eval_function  # get an evaluation function for a given model and dataset
    from deeplite_torch_zoo import create_model  # create a model with an arbitrary number of classes

.. _zoo_usage_load_dataset:

Loading Datasets
----------------

The loaded datasets are available as a dictionary of the following format: ``{'train': train_dataloder, 'test': test_dataloader}``. The `train_dataloder` and `test_dataloader` are objects of type ``torch.utils.data.DataLoader``.

.. _zoo_usage_load_dataset_classification:

Classification Datasets
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Example: DATASET_NAME = "cifar100", BATCH_SIZE = 128, MODEL_NAME = "resnet18"
    data_splits = get_data_splits_by_name(
        data_root="./",
        dataset_name=DATASET_NAME,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE
    )

.. _zoo_usage_load_dataset_od:

Object Detection Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^

The following sample code loads `PASCAL VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ dataset. ``train`` contains data loader for train sets for `VOC2007` and/or `VOC2012`. If both datasets are provided it concatenates both `VOC2007` and `VOC2012` train sets. Otherwise, it returns the train set for the provided dataset. 'test' contains dataloader (always with ``batch_size=1``) for test set based on `VOC2007`. You also need to provide the model name to instantiate the dataloaders.

.. code-block:: python

    # Example: DATASET_NAME = "voc", BATCH_SIZE = 32, MODEL_NAME = "yolo4s"
    data_splits = get_data_splits_by_name(
        data_root=PATH_TO_VOCdevkit,
        dataset_name=DATASET_NAME,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
    )

.. note::

    As it can be observed the dataloaders are provided based on the passed model name argument (`model_name`). Different object detection models consider inputs/outputs in different formats, and thus the `data_splits` are formatted according to the needs of the model.

.. _zoo_usage_load_models:

Loading Models
--------------

Models are generally provided with weights pretrained on specific datasets. One would load a model ``X`` pretrained on a dataset ``Y`` to get the appropriate weights for the task ``Y``. The ``get_model_by_name`` could used for this purpose. There is also an option to create a new model with an arbitrary number of categories for the downstream tasl and load the weights from another dataset for transfer learning (e.g. to load ``COCO`` weights to train a model on the ``VOC`` dataset). The ``create_model`` method should be generally used for that. Note that ``get_model_by_name`` always returns a fully-trained model for the specified task, this method thus does not allow specifying a custom number of classes.

.. _zoo_usage_load_models_classification:

Classification Models
^^^^^^^^^^^^^^^^^^^^^

To get a pretrained classification model one could use

.. code-block:: python

    model = get_model_by_name(
        model_name=MODEL_NAME, # example: "resnet18"
        dataset_name=DATASET_NAME, # example: "cifar100"
        pretrained=True, # or False, if pretrained weights are not required
        progress=False, # or True, if a progressbar is required
        device="cpu", # or "cuda"
    )

To create a new model with ImageNet weights and a custom number of classes one could use

.. code-block:: python

    model = create_model(
        model_name=MODEL_NAME, # example: "resnet18"
        pretraining_dataset=PRETRAIN_DATASET, # example: "imagenet"
        num_classes=NUM_CLASSES, # example: 42
        pretrained=True, # or False, if pretrained weights are not required
        progress=False, # or True, if a progressbar is required
        device="cpu", # or "cuda"
    )


This method would load the ImageNet-pretrained weights to all the modules of the model where one could match the shape of the weight tensors (i.e. all the layers except the last fully-connected one in the above case).

.. _zoo_usage_load_models_od:

Object Detection Models
^^^^^^^^^^^^^^^^^^^^^^^

To create an object detection model pretrained on a given dataset:

.. code-block:: python

    model = get_model_by_name(
        model_name=MODEL_NAME, # example: "yolo4s"
        dataset_name=DATASET_NAME, # example: "voc"
        pretrained=True, # or False, if pretrained weights are not required
        progress=False, # or True, if a progressbar is required
    )

Likewise, to create a object detection model with an arbitrary number of classes

.. code-block:: python

    model = create_model(
        model_name=MODEL_NAME, # example: "yolo4s"
        num_classes=NUM_CLASSES, # example: 8
        pretraining_dataset=PRETRAIN_DATASET, # example: "coco"
        pretrained=True, # or False, if pretrained weights are not required
        progress=False, # or True, if a progressbar is required
    )


Evaluating models
-----------------

To create an evaluation fuction for the given model and dataset one could call ``get_eval_function`` passing the ``model_name`` and ``dataset_name`` arguments:

.. code-block:: python

    eval_fn = get_eval_function(
        model_name=MODEL_NAME, # example: "resnet50"
        dataset_name=DATASET_NAME, # example: "imagenet"
    )


The returned evaluation function is a Python callable that takes two arguments: a PyTorch model object and a PyTorch dataloader object (logically corresponding to the test split dataloader) and returns a dictionary with metric names as keys and their corresponding values.

Please refer to the tables below for the performance metrics of the pretrained models available in the ``deeplite-torch-zoo``. After downloading the model, please evaluate the model using :ref:`profiler` to verify the metric values. However, one may see different numbers for the execution time as the target hardware and/or the load on the system may impact it.

.. _zoo_available_models:

Available Models
================

There is an useful utility function ``list_models`` which can be imported as

.. code-block:: python

    from deeplite_torch_zoo import list_models


This utility will help in listing available pretrained models or datasets.

For instance ``list_models("yolo5")`` will provide the list of available pretrained models that contain ``yolo5`` in their model names. Similar results e.g. can be obtained using ``list_models("yo")``. Filtering models by the corresponding task type is also possible by passing the string of the task type with the ``task_type_filter`` argument (the following task types are available: ``classification``, ``object_detection``, ``semantic_segmentation``).

.. code-block:: console

    +------------------+------------------------------------+
    | Available models |          Source datasets           |
    +==================+====================================+
    | yolo5_6l         | voc                                |
    +------------------+------------------------------------+
    | yolo5_6m         | coco, voc                          |
    +------------------+------------------------------------+
    | yolo5_6m_relu    | person_detection, voc              |
    +------------------+------------------------------------+
    | yolo5_6ma        | coco                               |
    +------------------+------------------------------------+
    | yolo5_6n         | coco, person_detection, voc, voc07 |
    +------------------+------------------------------------+
    | yolo5_6n_hswish  | coco                               |
    +------------------+------------------------------------+
    | yolo5_6n_relu    | coco, person_detection, voc        |
    +------------------+------------------------------------+
    | yolo5_6s         | coco, person_detection, voc, voc07 |
    +------------------+------------------------------------+
    | yolo5_6s_relu    | person_detection, voc              |
    +------------------+------------------------------------+
    | yolo5_6sa        | coco, person_detection             |
    +------------------+------------------------------------+
    | yolo5_6x         | voc                                |
    +------------------+------------------------------------+



.. _zoo_available_datasets:

Available Datasets
==================

+----+------------------------+--------------------+----------------------+------------+----------------------------------------+
| #  | Dataset (dataset_name) | Training Instances | Test Instances       | Resolution | Comments                               |
+----+------------------------+--------------------+----------------------+------------+----------------------------------------+
| 1  | MNIST                  | 60,000             | 10,000               | 28x28      | Downloadable through torchvision API   |
+----+------------------------+--------------------+----------------------+------------+----------------------------------------+
| 2  | CIFAR100               | 50,000             | 10,000               | 32x32      | Downloadable through torchvision API   |
+----+------------------------+--------------------+----------------------+------------+----------------------------------------+
| 3  | VWW                    | 40,775             | 8,059                | 224x224    | Based on COCO dataset                  |
+----+------------------------+--------------------+----------------------+------------+----------------------------------------+
| 4  | Tiny Imagenet          | 100,000            | 10,000               | 64x64      | Subset of Imagenet with 100 classes    |
+----+------------------------+--------------------+----------------------+------------+----------------------------------------+
| 5  | Imagenet10             | 385,244            | 15,011               | 224x224    | Subset of Imagenet2012 with 10 classes |
+----+------------------------+--------------------+----------------------+------------+----------------------------------------+
| 6  | Imagenet16             | 180,119            | 42,437               | 224x224    | Subset of Imagenet2012 with 16 classes |
+----+------------------------+--------------------+----------------------+------------+----------------------------------------+
| 7  | Imagenet               | 1,282,168          | 50,000               | 224x224    | Imagenet2012                           |
+----+------------------------+--------------------+----------------------+------------+----------------------------------------+
| 8  | VOC2007 (Detection)    | 5,011              | 4,952                | 500xH/Wx500| 20 classes, 24,640 annotated objects   |
+----+------------------------+--------------------+----------------------+------------+----------------------------------------+
| 9  | VOC2012 (Detection)    | 11,530 (train/val) | N/A                  | 500xH/Wx500| 20 classes, 27,450 annotated objects   |
+----+------------------------+--------------------+----------------------+------------+----------------------------------------+
| 10 | COCO2017 (Detection)   | 117,266, 5,000(val)| 40,670               | 300x300    | 80 Classes, 1.5M object instances      |
+----+------------------------+--------------------+----------------------+------------+----------------------------------------+
| 11 | COCO Person (Detection)| 39283(train/val)   | 1648                 | 300x300    | 1 Class                                |
+----+------------------------+--------------------+----------------------+------------+----------------------------------------+
.. _zoo_contribute:

Contribute a Model/Dataset to the Zoo
=====================================

Design
------

The ``deeplite-torch-zoo`` is organized as follows. It has two main directories: ``src`` and ``wrappers``. The ``src`` directory contains all the source code required to define and load the model and dataset. The ``wrappers`` contain the entry point API to load the dataset and model. The API definitions in the ``wrappers`` following a specific structure and any new model/dataset has to respect this structure.

.. code-block:: console

    - src
        - classification
        - objectdetection
        - segmentation
    - wrappers
        - datasets
            - classification
            - objectdetection
            - segmentation
        - models
            - classification
            - objectdetection
            - segmentation
        - eval

Contribute
----------

Please perform the following steps to contribute a new model or dataset to the ``deeplite-torch-zoo``

#. Add the source code under the following directory ``src/task_type``
    #. Add an existing repository as a `git-submodule <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`_
    #. otherwise, add the source code of data loaders, model definition, loss function, and eval function in a seperate directory
#. Train the model and upload the trained model weights in a public storage container. Please `contact us <support@deeplite.ai>`_ to add the trained model weights to Deeplite's common hosted `Amazon-S3` container.
#. Add API calls in ``wrappers`` directory:
    #. The entry point method for loading a model has to be named as: ``{model_name}_{dataset_name}_{num_classes}``
    #. The entry point method for dataloaders has to be named as: ``get_{dataset_name}_for_{model_name}``
    #. The `eval function` has to consider two inputs: (i) a model and (ii) a data_loader
#. Import the wrapper functions in the ``__init__`` file of the same directory
#. Add tests for the model in ``tests/real_tests/test_models.py`` check for the format in the file
#. Add fake test for the model in ``tests/fake_tests/test_models.py``

.. _zoo_benchmark_results:

Benchmark Results
=================

.. _zoo_benchmark_results_voc_od:

Models on VOC Object Detection Dataset
--------------------------------------

+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
| # | Architecture (model_name) |                                                                                                                                      | Size (MB) | MACs (Billions) | #Params (Millions) | Memory Footprint (MB)| Pretrained Weights                                                                                         |
|   |                           | `mean Average Precision <https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173>`_            |           |                 |                    |                      |                                                                                                            |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
| 1 | vgg16_ssd                 | 0.7733                                                                                                                               | 100.2731  | 31.4368         | 26.2860            | 309.7318             | `download <http://download.deeplite.ai/zoo/models/vgg16-ssd-voc-mp-0_7726-b1264e8beec69cbc.pth>`_          |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
| 2 | mb1_ssd                   | 0.6718                                                                                                                               | 36.1214   | 1.5547          | 9.4690             | 143.1124             | `download <http://download.deeplite.ai/zoo/models/mb1-ssd-voc-mp-0_675-58694caf.pth>`_                     |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
| 3 | resnet18_ssd              | 0.728                                                                                                                                | 32.489    | 6.2125          | 8.516              | 122.866              | `download <http://download.deeplite.ai/zoo/models/resnet18-ssd_voc_AP_0_728-564518d0c865972b.pth>`_        |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
| 4 | resnet34_ssd              | 0.761                                                                                                                                | 54.044    | 14.306          | 14.16              | 194.167              | `download <http://download.deeplite.ai/zoo/models/resnet34_ssd-voc_760-a102a7ca6564ab44.pth>`_             |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
| 5 | resnet50_ssd              | 0.766                                                                                                                                | 58.853    | 16.2557         | 15.428             | 443.1532             | `download <http://download.deeplite.ai/zoo/models/resnet50_ssd-voc_766-d934cbe063398fcd.pth>`_             |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
| 6 | mb2_ssd_lite              | 0.687                                                                                                                                | **12.9**  | 0.699           | 3.38               | 149.7                | `download <http://download.deeplite.ai/zoo/models/mb2-ssd-lite-voc-mp-0_686-b0d1ac2c.pth>`_                |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
| 7 | yolo-v3                   | 0.8291                                                                                                                               | 235.0847  | 38.0740         | 61.6260            | 999.7075             | `download <http://download.deeplite.ai/zoo/models/yolo3-voc-0_839-a6149826183808aa.pth>`_                  |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
| 8 | yolo-v4s                  | 0.849                                                                                                                                | 34.9      | 5.1             | 9.1                | 355.72               | `download <http://download.deeplite.ai/zoo/models/yolov4s-voc-20classes_849_58041e8852a4b2e2.pt>`_         |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
| 9 | yolo-v4m                  | 0.874                                                                                                                                | 93.2      | 13              | 24.4               | 606.41               | `download <http://download.deeplite.ai/zoo/models/yolov4m-voc-20classes_874_e0c8e179992b5da2.pt>`_         |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
|10 | yolo-v4l                  | 0.872                                                                                                                                | 200.65    | 29.30           | 52.60              | 1006.24              | `download <http://download.deeplite.ai/zoo/models/yolo4l-voc-20classes_872-9f54132ce2934fbf.pth>`_         |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
|11 | yolo-v4l-leaky            | 0.891                                                                                                                                | 200.65    | 29.35           | 52.60              | 1006.24              | `download <http://download.deeplite.ai/zoo/models/yolo4l-leaky-voc-20classes_891-2c0f78ee3938ade3.pt>`_    |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
|12 | yolo-v4x                  | 0.882                                                                                                                                | 368       | 55.32           | 96                 | 1528                 | `download <http://download.deeplite.ai/zoo/models/yolo4x-voc-20classes_882-187f352b9d0d29c6.pth>`_         |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
|13 | yolo-v5l                  | 0.875                                                                                                                                | 176.39    | 26.52           | 46.24              | 806.64               | `download <http://download.deeplite.ai/zoo/models/yolov5_6l-voc-20classes_875_3fb90f0c405f170c.pt>`_       |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
|14 | yolo-v5m                  | 0.902                                                                                                                                | 79.91     | 11.82           | 20.94              | 471.96               | `download <http://download.deeplite.ai/zoo/models/yolo5_6m-voc-20classes_902-50c151baffbf896e.pt>`_        |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
|15 | yolo-v5m-relu             | 0.856                                                                                                                                | 79.91     | 11.85           | 20.94              | 471.9                | `download <http://download.deeplite.ai/zoo/models/yolov5_6m_relu-voc-20classes-856_c5c23135e6d5012f.pt>`_  |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
|18 | yolo-v5n                  | 0.762                                                                                                                                | 6.832     | 1.043           | 1.790              | 115.40               | `download <http://download.deeplite.ai/zoo/models/yolo5_6n-voc-20classes_762-a6b8573a32ebb4c8.pt>`_        |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
|19 | yolo-v5s                  | 0.871                                                                                                                                | 26.98     | 3.92            | 7.073              | 235.95               | `download <http://download.deeplite.ai/zoo/models/yolo5_6s-voc-20classes_871-4ceb1b22b227c05c.pt>`_        |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
|21 | yolo-v5s_relu             | 0.819                                                                                                                                | 26.98     | 3.93            | 7.073              | 235.95               | `download <http://download.deeplite.ai/zoo/models/yolov5_6s_relu-voc-20classes-819_a35dff53b174e383.pt>`_  |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
|22 | yolo-v5x                  | 0.884                                                                                                                                | 329.4     | 50.13           | 86.34              | 1252.96              | `download <http://download.deeplite.ai/zoo/models/yolov5_6x-voc-20classes_884_a2b6fb7234218cf6.pt>`_       |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+

.. _zoo_benchmark_results_coco_od:

Models on COCO Object Detection Dataset
--------------------------------------

+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+-------------------------------------------------------------------------------------------------------------+
| # | Architecture (model_name) |                                                                                                                                      | Size (MB) | MACs (Billions) | #Params (Millions) | Memory Footprint (MB)| Pretrained Weights                                                                                          |
|   |                           | `mean Average Precision <https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173>`_            |           |                 |                    |                      |                                                                                                             |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+-------------------------------------------------------------------------------------------------------------+
| 1 | yolo4m                    | 0.309                                                                                                                                | 94.133    | 11.44           | 24.67              | 548.83               | `download <http://download.deeplite.ai/zoo/models/yolov4_6m-coco-80classes-309_02b2013002a4724b.pt>`_       |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+-------------------------------------------------------------------------------------------------------------+
| 2 | yolo4s                    | 0.288                                                                                                                                | 35.58     | 4.50            | 9.32               | 324.34               | `download <http://download.deeplite.ai/zoo/models/yolov4_6s-coco-80classes-288_b112910223d6c56d.pt>`_       |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+-------------------------------------------------------------------------------------------------------------+
| 3 | yolo5_6m                  | 0.374                                                                                                                                | 80.83     | 10.36           | 21.19              | 431.063              | `download <http://download.deeplite.ai/zoo/models/yolov5_6m-coco-80classes_374-f93fa94b629c45ab.pt>`_       |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+-------------------------------------------------------------------------------------------------------------+
| 4 | yolo5_6n                  | 0.211                                                                                                                                | 7.14      | 0.954           | 1.87               | 112.94               | `download <http://download.deeplite.ai/zoo/models/yolov5_6n-coco-80classes_211-e9e44a7de1f08ea2.pt>`_       |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+-------------------------------------------------------------------------------------------------------------+
| 5 | yolo5_6n_hswish           | 0.183                                                                                                                                | 7.14      | 0.954           | 1.872              | 112.94               | `download <http://download.deeplite.ai/zoo/models/yolov5_6n_hswish-coco-80classes-183-a2fed163ec98352a.pt>`_|
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+-------------------------------------------------------------------------------------------------------------+
| 6 | yolo5_6n_relu             | 0.167                                                                                                                                | 34.9      | 5.1             | 9.1                | 320.7                | `download <http://download.deeplite.ai/zoo/models/yolov5_6n_relu-coco-80classes-167-7b6609497c63df79.pt>`_  |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+-------------------------------------------------------------------------------------------------------------+
| 7 | yolo5_6s                  | 0.301                                                                                                                                | 27.60     | 3.49            | 7.235              | 219.96               | `download <http://download.deeplite.ai/zoo/models/yolov5_6s-coco-80classes_301-8ff1dabeec225366.pt>`_       |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+-------------------------------------------------------------------------------------------------------------+

.. _zoo_benchmark_results_coco_person_od:

Models on COCO Person Detection Dataset (only `person` class from the 80-class COCO dataset)
----------------------------------------------
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------------------+
| # | Architecture (model_name) |                                                                                                                                      | Size (MB) | MACs (Billions) | #Params (Millions) | Memory Footprint (MB)| Pretrained Weights                                                                                                     |
|   |                           | `mean Average Precision <https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173>`_            |           |                 |                    |                      |                                                                                                                        |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------------------+
| 1 | yolo5_6m_relu             | 0.709                                                                                                                                | 79.61     | 6.015           | 20.87              | 277.36               | `download <http://download.deeplite.ai/zoo/models/yolov5_6m_relu-person-detection-1class_709-3f59321c540d2d1c.pt>`_    |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------------------+
| 2 | yolo5_6n                  | 0.6718                                                                                                                               | 6.73      | 0.522011        | 1.765              | 59.84                | `download <http://download.deeplite.ai/zoo/models/yolov5_6n-person-detection-1class_696-fff2a2c720e20752.pt>`_         |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------------------+
| 3 | yolo5_6n_relu             | 0.621                                                                                                                                | 6.73      | 0.5249          | 1.765              | 59.847               | `download <http://download.deeplite.ai/zoo/models/yolov5_6n_relu-person-detection-1class_621-6794298f12d33ba8.pt>`_    |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------------------+
| 4 | yolo5_6s                  | 0.738                                                                                                                                | 26.788    | 1.981           | 7.022              | 131.122              | `download <http://download.deeplite.ai/zoo/models/yolov5_6s-person-detection-1class_738-9e9ac9dae14b0dcd.pt>`_         |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------------------+
| 5 | yolo5_6s_relu             | 0.682                                                                                                                                | 26.788    | 1.987           | 7.022              | 131.122              | `download <http://download.deeplite.ai/zoo/models/yolov5_6s_relu-person-detection-1class_682-45ae979a06b80767.pt>`_    |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------------------+
| 6 | yolo5_6sa                 | 0.659                                                                                                                                | 47        | 2.026           | 12.32              | 153.033              | `download <http://download.deeplite.ai/zoo/models/yolov5_6sa-person-detection-1class_659_015807ae6899af0f.pt>`_        |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------------------+

.. _zoo_benchmark_results_voc07_od:

Models on VOC2007 Dataset (VOC2007 train split taken as training data and VOC2007 val split used for testing)
----------------------------------------------
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------------------+
| # | Architecture (model_name) |                                                                                                                                      | Size (MB) | MACs (Billions) | #Params (Millions) | Memory Footprint (MB)| Pretrained Weights                                                                                                     |
|   |                           | `mean Average Precision <https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173>`_            |           |                 |                    |                      |                                                                                                                        |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------------------+
| 1 | yolo5_6n                  | 0.620                                                                                                                                | 6.83      | 1.043           | 1.79               | 115.40               | `download <http://download.deeplite.ai/zoo/models/yolov5_6n-voc07-20classes-620_037230667eff7b12.pt>`_                 |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------------------+
| 2 | yolo5_6s                  | 0.687                                                                                                                                | 26.98     | 3.92            | 7.07               | 235.95               | `download <http://download.deeplite.ai/zoo/models/yolov5_6s-voc07-20classes-687_4d221fd4edc09ce1.pt>`_                 |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------------------+


.. _zoo_benchmark_results_voc_seg:

Models on VOC Segmentation Dataset
----------------------------------

+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
| # | Architecture (model_name) |                                                                                                                                      | Size (MB) | MACs (Billions) | #Params (Millions) | Memory Footprint (MB)| Pretrained Weights                                                                                         |
|   |                           | `mean Inter. over Union`                                                                                                             |           |                 |                    |                      |                                                                                                            |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
| 1 | unet_scse_resnet18        | 0.582                                                                                                                                | 83.3697   | 20.8930         | 21.8549            | 575.0954             | `download <http://download.deeplite.ai/zoo/models/unet_scse_resnet18-voc-miou_593-1e0987c833e9abd7.pth>`_  |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
| 2 | unet_scse_resnet18_1cls   | 0.673                                                                                                                                | 83.3647   | 20.5522         | 21.8536            | 535.0954             | `download <http://download.deeplite.ai/zoo/models/unet_scse_resnet18-voc-1cls-0_682-38cbf3aaa2ce9a46.pth>`_|
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
| 3 | unet_scse_resnet18_2cls   | 0.679                                                                                                                                | 83.3652   | 20.5862         | 21.8537            | 539.0954             | `download <http://download.deeplite.ai/zoo/models/unet_scse_resnet18-voc-2cls-0_688-79087739621c42c1.pth>`_|
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
| 4 | fcn32                     | 0.713                                                                                                                                | 519.382   | 136.142         | 136.152            | 858.2010             | `download <http://download.deeplite.ai/zoo/models/fcn32-voc-20_713-b745bd7e373e31d1.pth>`_                 |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+
| 5 | deeplab_mobilenet         | 0.571                                                                                                                                | 29.0976   | 26.4870         | 5.8161             | 1134.6057            | `download <http://download.deeplite.ai/zoo/models/deeplab-mobilenet-voc-20_593-94ac51da679409d6.pth>`_     |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------------+

.. _zoo_benchmark_results_mnist:

Models on MNIST dataset
-----------------------

+---+---------------------------+---------+----------+-----------------+--------------------+----------------------+---------------------------------------------------------------------------------------+
| # | Architecture (model_name) | Top1 (%)| Size (MB)| MACs (Millions) | #Params (Millions) | Memory Footprint (MB)| Pretrained Weights                                                                    |
+---+---------------------------+---------+----------+-----------------+--------------------+----------------------+---------------------------------------------------------------------------------------+
| 1 | lenet5                    | 99.1199 | 0.1695   | 0.2930          | 0.0444             | 0.1904               | `download <http://download.deeplite.ai/zoo/models/lenet-mnist-e5e2d99e08460491.pth>`_ |
+---+---------------------------+---------+----------+-----------------+--------------------+----------------------+---------------------------------------------------------------------------------------+
| 2 | mlp2                      | 97.8046 | 0.4512   | 0.1211          | 0.1183             | 0.4572               | `download <http://download.deeplite.ai/zoo/models/mlp2-mnist-cd7538f979ca4d0e.pth>`_  |
+---+---------------------------+---------+----------+-----------------+--------------------+----------------------+---------------------------------------------------------------------------------------+
| 3 | mlp4                      | 97.8145 | 0.5772   | 0.1549          | 0.1513             | 0.5861               | `download <http://download.deeplite.ai/zoo/models/mlp4-mnist-c6614ff040df60a4.pth>`_  |
+---+---------------------------+---------+----------+-----------------+--------------------+----------------------+---------------------------------------------------------------------------------------+
| 4 | mlp8                      | 96.6970 | 0.8291   | 0.2226          | 0.2174             | 0.8439               | `download <http://download.deeplite.ai/zoo/models/mlp8-mnist-de6f135822553043.pth>`_  |
+---+---------------------------+---------+----------+-----------------+--------------------+----------------------+---------------------------------------------------------------------------------------+

.. _zoo_benchmark_results_cifar100:

Models on CIFAR100 dataset
----------------------------

+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------+
| #  | Architecture (model_name) | Top1 (%) | Size (MB) | MACs (Billions) | #Params (Millions) | Memory Footprint (MB)| Pretrained Weights                                                                                   |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------+
| 1  | resnet18                  | 76.8295  | 42.8014   | 0.5567          | 11.2201            | 48.4389              | `download <http://download.deeplite.ai/zoo/models/resnet18-cifar100-86b0c368c511bd57.pth>`_          |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------+
| 2  | resnet50                  | 78.0657  | 90.4284   | 1.3049          | 23.7053            | 123.5033             | `download <http://download.deeplite.ai/zoo/models/resnet50-cifar100-d03f14e3031410de.pth>`_          |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------+
| 3  | vgg19                     | 72.3794  | 76.6246   | 0.3995          | 20.0867            | 80.2270              | `download <http://download.deeplite.ai/zoo/models/vgg19-cifar100-6d791de492a133b6.pth>`_             |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------+
| 4  | densenet121               | 78.4612  | 26.8881   | 0.8982          | 7.0485             | 66.1506              | `download <http://download.deeplite.ai/zoo/models/densenet121-cifar100-7e4ec64b17b04532.pth>`_       |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------+
| 5  | googlenet                 | 79.3513  | 23.8743   | 1.5341          | 6.2585             | 64.5977              | `download <http://download.deeplite.ai/zoo/models/googlenet-cifar100-15f970a22f56433f.pth>`_         |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------+
| 6  | mobilenet_v1              | 66.8414  | 12.6246   | 0.0473          | 3.3095             | 16.6215              | `download <http://download.deeplite.ai/zoo/models/mobilenetv1-cifar100-4690c1a2246529eb.pth>`_       |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------+
| 7  | mobilenet_v2              | 73.0815  | 9.2019    | 0.0947          | 2.4122             | 22.8999              | `download <http://download.deeplite.ai/zoo/models/mobilenetv2-cifar100-a7ba34049d626cf4.pth>`_       |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------+
| 8  | pre_act_resnet18          | 76.5229  | 42.7907   | 0.5566          | 11.2173            | 48.1781              | `download <http://download.deeplite.ai/zoo/models/pre_act_resnet18-cifar100-1c4d1dc76ee9c6f6.pth>`_  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------+
| 9  | resnext29_2x64d           | 79.9150  | 35.1754   | 1.4167          | 9.2210             | 67.6879              | `download <http://download.deeplite.ai/zoo/models/resnext29_2x64d-cifar100-f6ba33baf30048d1.pth>`_   |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------+
| 10 | shufflenet_v2_1_0         | 69.9169  | 5.1731    | 0.0462          | 1.356              | 12.3419              | `download <http://download.deeplite.ai/zoo/models/shufflenet_v2_l.0-cifar100-16ae6f50f5adecad.pth>`_ |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+------------------------------------------------------------------------------------------------------+

.. _zoo_benchmark_results_vww:

Models on VWW dataset
---------------------

+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+-------------------------------------------------------------------------------------------------------+
| # | Architecture (model_name) | Top1 (%) | Size (MB) | MACs (Billions) | #Params (Millions) | Memory Footprint (MB)| Pretrained Weights                                                                                    |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+-------------------------------------------------------------------------------------------------------+
| 1 | resnet18                  | 93.5496  | 42.6389   | 1.8217          | 11.1775            | 74.6057              | `download <http://download.deeplite.ai/zoo/models/resnet18-vww-7f02ab4b50481ab7.pth>`_                |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+-------------------------------------------------------------------------------------------------------+
| 2 | resnet50                  | 94.3675  | 89.6917   | 4.1199          | 23.5121            | 233.5413             | `download <http://download.deeplite.ai/zoo/models/resnet50-vww-9d4cb2cb19f8c5d5.pth>`_                |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+-------------------------------------------------------------------------------------------------------+
| 3 | mobilenet_v1              | 92.4444  | 12.2415   | 0.5829          | 3.2090             | 70.5286              | `download <http://download.deeplite.ai/zoo/models/mobilenetv1-vww-84f65dc4bc649cd6.pth>`_             |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+-------------------------------------------------------------------------------------------------------+
| 3 | mobilenet_v3_small        | 89.1180  | 5.7980    | 0.0599          | 1.5199             | 30.2576              | `download <http://download.deeplite.ai/zoo/models/mobilenetv3-small-vww-89_20-5224256355d8fbfa.pth>`_ |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+-------------------------------------------------------------------------------------------------------+
| 3 | mobilenet_v3_large        | 89.1800  | 16.0393   | 0.2286          | 4.2046             | 83.8590              | `download <http://download.deeplite.ai/zoo/models/mobilenetv3-large-vww-89_14-e80487ebdbb41d5a.pth>`_ |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+-------------------------------------------------------------------------------------------------------+

.. _zoo_benchmark_results_imagenet10:

Models on Imagenet10 dataset
----------------------------

+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+----------------------------------------------------------------------------------------+
| # | Architecture (model_name) | Top1 (%) | Size (MB) | MACs (Billions) | #Params (Millions) | Memory Footprint (MB)| Pretrained Weights                                                                     |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+----------------------------------------------------------------------------------------+
| 1 | resnet18                  | 93.8294  | 42.6546   | 1.8217          | 11.1816            | 74.6215              | `download <http://download.deeplite.ai/zoo/models/resnet18-vww-7f02ab4b50481ab7.pth>`_ |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+----------------------------------------------------------------------------------------+
| 2 | mobilenet_v2_0_35         | 81.0492  | 1.5600    | 0.0664          | 0.4089             | 34.9010              | `download <http://download.deeplite.ai/zoo/models/resnet50-vww-9d4cb2cb19f8c5d5.pth>`_ |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+----------------------------------------------------------------------------------------+

.. _zoo_benchmark_results_imagenet16:

Models on Imagenet16 dataset
----------------------------

+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+----------------------------------------------------------------------------------------+
| # | Architecture (model_name) | Top1 (%) | Size (MB) | MACs (Billions) | #Params (Millions) | Memory Footprint (MB)| Pretrained Weights                                                                     |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+----------------------------------------------------------------------------------------+
| 1 | resnet18                  | 94.5115  | 42.6663   | 1.8217          | 11.1816            | 74.6332              | `download <http://download.deeplite.ai/zoo/models/resnet18-vww-7f02ab4b50481ab7.pth>`_ |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+----------------------------------------------------------------------------------------+
| 2 | resnet50                  | 96.8518  | 89.8011   | 4.1199          | 23.5408            | 233.6508             | `download <http://download.deeplite.ai/zoo/models/resnet50-vww-9d4cb2cb19f8c5d5.pth>`_ |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+----------------------------------------------------------------------------------------+

.. _zoo_benchmark_results_imagenet:

Models on Imagenet dataset (from torchvision)
---------------------------------------------

+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+
| #  | Architecture (model_name) | Top1 (%) | Size (MB) | MACs (Billions) | #Params (Millions) | Memory Footprint (MB)| Pretrained Weights |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+
| 1  | resnet18                  | 69.7319  | 44.5919   | 1.8222          | 11.6895            | 76.5664              | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+
| 2  | resnet34                  | 73.2880  | 83.1515   | 3.6756          | 21.7977            | 131.8740             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+
| 3  | resnet50                  | 76.1001  | 97.4923   | 4.1219          | 25.5570            | 241.3496             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+
| 4  | resnet101                 | 77.3489  | 169.9416  | 7.8495          | 44.549             | 385.3847             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+
| 5  | resnet152                 | 78.2836  | 229.6173  | 11.5807         | 60.1928            | 533.4902             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+
| 6  | inception_v3              | 69.5109  | 90.9217   | 2.8472          | 27.1613            |  149.3052            | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+
| 7  | densenet121               | 74.4106  | 30.4369   | 2.8826          | 7.9789             | 187.7805             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+
| 8  | densenet161               | 77.1120  | 109.4093  | 7.8184          | 28.681             | 393.9603             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+
| 9  | densenet169               | 75.5635  | 53.9760   | 3.4184          | 14.149             | 238.9538             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+
| 10 | densenet201               | 76.8702  | 76.3471   | 4.3670          | 20.0139            | 307.5974             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+
| 11 | alexnet                   | 56.4758  | 233.0812  | 0.7156          | 61.1008            | 237.8486             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+
| 12 | squeezenet1_0             | 58.0591  | 4.7624    | 0.8300          | 1.2484             | 51.2403              | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+
| 13 | squeezenet1_1             | 58.1438  | 4.7130    | 0.3559          | 1.235              | 32.1729              | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+
| 14 | vgg11                     | 68.9946  | 506.8334  | 7.6301          | 132.8633           | 570.0989             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+
| 15 | vgg11_bn                  | 70.3433  | 506.8544  | 7.6449          | 132.8688           | 598.4480             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+
| 16 | vgg13                     | 69.9017  | 507.5373  | 11.3391         | 133.0478           | 607.5527             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+
| 17 | vgg13_bn                  | 71.5557  | 507.5597  | 11.3636         | 133.0537           | 654.2783             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+
| 18 | vgg16                     | 71.5605  | 527.7921  | 15.5035         | 138.3575           | 637.7607             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+
| 19 | vgg16_bn                  | 73.3352  | 527.8243  | 15.5306         | 138.3660           | 689.4726             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+
| 20 | vgg19                     | 72.3449  | 548.0470  | 19.6679         | 143.6672           | 667.9687             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+
| 21 | vgg19_bn                  | 74.1900  | 548.0890  | 19.6976         | 143.6782           | 724.6669             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+

Models on Imagenet dataset (timm/torchvision)
---------------------------------------------

The zoo enables to load any ImageNet-pretrained model from the `timm repo <https://github.com/rwightman/pytorch-image-models>`_ as well as any ImageNet model from torchvision. In case the model names overlap with timm, the corresponding timm model is loaded.

- **Model Size:** Memory consumed by the parameters (weights and biases) of the model
- **MACs:** Summation of Multiply-Add Cumulations (MACs) per single image (batch_size=1)
- **#Parameters:** Total number of parameters (trainable and non-trainable) in the model
- **Memory Footprint:** Total memory consumed by the parameters (weights and biases) and activations (per layer) per single image (batch_size=1)
