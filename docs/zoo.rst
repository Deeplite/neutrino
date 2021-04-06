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

- numpy==1.18.5
- torch==1.4.0
- torchvision==0.5.0
- opencv-python
- scipy>=1.4.1
- pycocotools
- Cython==0.28.4
- scikit-image==0.15.0
- tqdm==4.46.0
- albumentations==0.1.8
- pretrainedmodels==0.7.4
- torchfcn
- tensorboardX
- mmcv==1.2.0
- xtcocotools>=1.6
- json-tricks>=3.15.4
- poseval@git+https://github.com/svenkreiss/poseval.git#egg=poseval-0.1.0
- black
- isort

.. _zoo_usage:

How to Use
==========

The ``deeplite-torch-zoo`` is collection of benchmark computer vision datasets and pretrained models. There are two primary wrapper functions to load datasets and models, ``get_data_splits_by_name``, ``get_model_by_name`` (available in ``deeplite_torch_zoo.wrappers.wrapper``)

.. _zoo_usage_load_dataset:

Loading Datasets
----------------

The loaded datasets are available as a dictionary of the following format: ``{'train': train_dataloder, 'test': test_dataloader}``. The `train_dataloder` and `test_dataloader` are objects of ``torch.utils.data.DataLoader``.

.. _zoo_usage_load_dataset_classification:

Classification Datasets
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    
    # Example: DATASET_NAME = "cifar100", BATCH_SIZE = 128
    data_splits = get_data_splits_by_name(
        dataset_name=DATASET_NAME, batch_size=BATCH_SIZE
    )

.. _zoo_usage_load_dataset_od:

Object Detection Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^

The following sample code loads `PASCAL VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ dataset. ``train`` contains data loader for train sets for `VOC2007` and/or `VOC2012`. If both datasets are provided it concatenates both `VOC2007` and `VOC2012` train sets. Otherwise, it returns the train set for the provided dataset. 'test' contains dataloader (always with ``batch_size=1``) for test set based on `VOC2007`. You also need to provide the model type as well.

.. code-block:: python

    data_splits = get_data_splits_by_name(
        data_root=PATH_TO_VOCdevkit,
        dataset_name="voc",
        model_name="vgg16_ssd",
        batch_size=BATCH_SIZE,
    )

.. note::

    As it can be observed the data_loaders are provided based on the corresponding model (`model_name`). Different object detection models consider inputs/outputs in different formats, and thus the `data_splits` are formatted according to the needs of the model.

.. _zoo_usage_load_models:

Loading Models
--------------

Models are provided with pretrained weights on specific datasets. Thus, one could load a model ``X`` pretrained on dataset ``Y``, for getting the appropriate weights. 

.. _zoo_usage_load_models_classification:

Classification Models
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    
    model = get_model_by_name(
        model_name=MODEL_NAME, # example: "resnet18"
        dataset_name=DATASET_NAME, # example: "cifar100"
        pretrained=True, # or False, if pretrained weights are not required
        progress=False, # or True, if a progressbar is required
        device="cpu", # or "gpu"
    )

.. _zoo_usage_load_models_od:

Object Detection Models
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    model = get_model_by_name(
        model_name=MODEL_NAME, # example: "vgg16_ssd"
        dataset_name=DATASET_NAME, # example: "voc_20"
        pretrained=True, # or False, if pretrained weights are not required
        progress=False, # or True, if a progressbar is required
    )

To evaluate a model, the following style of code could be used,

.. code-block:: python
    
    test_loader = data_splits["test"]
    APs = vgg16_ssd_eval_func(model, test_loader)


Please refer to the tables below for the performance metrics of the pretrained models available in the ``deeplite-torch-zoo``. After downloading the model, please evaluate the model using :ref:`profiler` to verify the metric values. However, one may see different numbers for the execution time as the target hardware and/or the load on the system may impact it.

.. _zoo_available_models:

Available Models
================

There is an important utility function ``list_models`` (available in ``deeplite_torch_zoo.wrappers.wrapper``). This utility will help in listing all available pretrained models or datasets.

For instance ``list_models("yolo3")`` will provide the following result. Similar results can be obtained using ``list_models("yo")``.

.. code-block:: console

    yolo3
    yolo3_voc_1
    yolo3_voc_2
    yolo3_voc_6
    yolo3_voc_20
    yolo3_lisa_11


.. _zoo_available_datasets:

Available Datasets
==================

+---+------------------------+--------------------+----------------------+------------+----------------------------------------+----------------------------------------------------------------------------------+
| # | Dataset (dataset_name) | Training Instances | Test Instances       | Resolution | Comments                               | download                                                                         |
+---+------------------------+--------------------+----------------------+------------+----------------------------------------+----------------------------------------------------------------------------------+
| 1 | MNIST                  | 60,000             | 10,000               | 28x28      | Downloadable through torchvision API   | N/A                                                                              |
+---+------------------------+--------------------+----------------------+------------+----------------------------------------+----------------------------------------------------------------------------------+
| 2 | CIFAR100               | 50,000             | 10,000               | 32x32      | Downloadable through torchvision API   | N/A                                                                              |
+---+------------------------+--------------------+----------------------+------------+----------------------------------------+----------------------------------------------------------------------------------+
| 3 | VWW                    | 40,775             | 8,059                | 224x224    | Based on COCO dataset                  | `download <https://drive.google.com/open?id=15CP_uWUoj-p-CGq594v0iclU2MR17lrf>`_ |
+---+------------------------+--------------------+----------------------+------------+----------------------------------------+----------------------------------------------------------------------------------+
| 4 | Imagenet10             | 385,244            | 15,011               | 224x224    | Subset of Imagenet2012 with 10 classes | `download <https://drive.google.com/open?id=1KXdv-S4AvwtcF8-yj2klwDB44A4gzLKG>`_ |
+---+------------------------+--------------------+----------------------+------------+----------------------------------------+----------------------------------------------------------------------------------+
| 5 | Imagenet16             | 180,119            | 42,437               | 224x224    | Subset of Imagenet2012 with 16 classes | `download <https://drive.google.com/open?id=1c-LoMwGKNdiM0-Of8D4Wjyds-HpG7OLe>`_ |
+---+------------------------+--------------------+----------------------+------------+----------------------------------------+----------------------------------------------------------------------------------+
| 6 | Imagenet               | 1,282,168          | 50,000               | 224x224    | Imagenet2012                           | `download <https://drive.google.com/open?id=15T4v_kvau0P08kuwufCTqeujzkDfVdRr>`_ |
+---+------------------------+--------------------+----------------------+------------+----------------------------------------+----------------------------------------------------------------------------------+
| 7 | VOC2007 (Detection)    | 5,011              | 4,952                | 500xH/Wx500| 20 classes, 24,640 annotated objects   | `download <https://drive.google.com/open?id=1Isvu0qMMzOUojWeRzNJ-PkM9bAvGYRFp>`_ |
+---+------------------------+--------------------+----------------------+------------+----------------------------------------+----------------------------------------------------------------------------------+
| 8 | VOC2012 (Detection)    | 11,530 (train/val) | N/A                  | 500xH/Wx500| 20 classes, 27,450 annotated objects   | `download <https://drive.google.com/open?id=1o6wsXsG3yFXeYuzN9_pi13_-4JmouiQx>`_ |
+---+------------------------+--------------------+----------------------+------------+----------------------------------------+----------------------------------------------------------------------------------+
| 9 | COCO2017 (Detection)   | 117,266, 5,000(val)| 40,670               | 300x300    | 80 Classes, 1.5M object instances      | `download <https://drive.google.com/open?id=1WD5fVHWQFE0cHp28P9eI2dyEXV_Mj5lw>`_ |
+---+------------------------+--------------------+----------------------+------------+----------------------------------------+----------------------------------------------------------------------------------+

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

+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------------+
| # | Architecture (model_name) |                                                                                                                                      | Size (MB) | MACs (Billions) | #Params (Millions) | Memory Footprint(MB) | Execution Time(ms) | Pretrained Weights                                                                                         |
|   |                           | `mean Average Precision <https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173>`_            |           |                 |                    |                      |                    |                                                                                                            |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------------+
| 1 | vgg16_ssd                 | 0.7733                                                                                                                               | 100.2731  | 31.4368         | 26.2860            | 309.7318             | 3.8033             | `download <http://download.deeplite.ai/zoo/models/vgg16-ssd-voc-mp-0_7726-b1264e8beec69cbc.pth>`_          |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------------+
| 2 | mb1_ssd                   | 0.6718                                                                                                                               | 36.1214   | 1.5547          | 9.4690             | 143.1124             | 4.6199             | `download <http://download.deeplite.ai/zoo/models/mb1-ssd-voc-mp-0_675-58694caf.pth>`_                     |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------------+
| 3 | ssd-resnet18              | 0.580                                                                                                                                | 32.489    | 6.2125          | 8.516              | 83.3148              | 1.1355             | `download <http://download.deeplite.ai/zoo/models/ssd300-resnet18-voc20classes_580-cfc94e5b701953ba.pth>`_ |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------------+
| 4 | ssd-resnet34              | 0.654                                                                                                                                | 54.044    | 14.306          | 14.16              | 142.284              | 2.1151             | `download <http://download.deeplite.ai/zoo/models/ssd300-resnet34-voc20classes_654-eafd64758f6bfd1d.pth>`_ |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------------+
| 5 | ssd-resnet50              | 0.659                                                                                                                                | 58.853    | 16.2557         | 15.428             | 407.3344             | 4.1539             | `download <http://download.deeplite.ai/zoo/models/ssd300-resnet50-voc20classes_659-07069cb099a9a8b8.pth>`_ |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------------+
| 6 | ssd-vgg16                 | 0.641                                                                                                                                | 95.8805   | 31.6305         | 25.13              | 120.3495             | 4.8609             | `download <http://download.deeplite.ai/zoo/models/ssd300-vgg16-voc20classes_641-07cc9e5fecdcecc1.pth>`_    |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------------+
| 7 | mb2_ssd_lite              | 0.687                                                                                                                                | **12.9**  | 0.699           | 3.38               | 149.7                | 1.22               | `download <http://download.deeplite.ai/zoo/models/mb2-ssd-lite-voc-mp-0_686-b0d1ac2c.pth>`_                |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------------+
| 8 | yolo-v3                   | 0.8291                                                                                                                               | 235.0847  | 38.0740         | 61.6260            | 999.7075             | 16.8699            | `download <http://download.deeplite.ai/zoo/models/yolo3-voc-0_839-a6149826183808aa.pth>`_                  |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------------+
| 9 | yolo-v4s                  | 0.857                                                                                                                                | 34.9      | 5.1             | 9.1                | 320.7                | 3.4                | `download <http://download.deeplite.ai/zoo/models/yolo4s-voc-20classes_850-270ddc5d43290a95.pth>`_         |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------------+
|10 | yolo-v4m                  | 0.882                                                                                                                                | 93.2      | 13              | 24.4               | 513.2                | 6.2                | `download <http://download.deeplite.ai/zoo/models/yolo4m-voc-20classes_885-b854caad9ca7fb7c.pth>`_         |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------------+
|11 | yolo-v4l                  | 0.882                                                                                                                                | 200       | 29              | 52                 | 805                  | 17                 | `download <http://download.deeplite.ai/zoo/models/yolo4l-voc-20classes_872-9f54132ce2934fbf.pth>`_         |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------------+
|12 | yolo-v4x                  | 0.893                                                                                                                                | 368       | 55              | 96                 | 1159                 | 16                 | `download <http://download.deeplite.ai/zoo/models/yolo4x-voc-20classes_882-187f352b9d0d29c6.pth>`_         |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------------+

.. _zoo_benchmark_results_voc_seg:

Models on VOC Segmentation Dataset 
----------------------------------

+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------------+
| # | Architecture (model_name) |                                                                                                                                      | Size (MB) | MACs (Billions) | #Params (Millions) | Memory Footprint(MB) | Execution Time(ms) | Pretrained Weights                                                                                         |
|   |                           | `mean Inter. over Union`                                                                                                             |           |                 |                    |                      |                    |                                                                                                            |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------------+
| 1 | unet_scse_resnet18        | 0.582                                                                                                                                | 83.3697   | 20.8930         | 21.8549            | 575.0954             | 9.6361             | `download <http://download.deeplite.ai/zoo/models/unet_scse_resnet18-voc-miou_593-1e0987c833e9abd7.pth>`_  |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------------+
| 2 | unet_scse_resnet18_1cls   | 0.673                                                                                                                                | 83.3647   | 20.5522         | 21.8536            | 535.0954             | 9.4987             | `download <http://download.deeplite.ai/zoo/models/unet_scse_resnet18-voc-1cls-0_682-38cbf3aaa2ce9a46.pth>`_|
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------------+
| 3 | unet_scse_resnet18_2cls   | 0.679                                                                                                                                | 83.3652   | 20.5862         | 21.8537            | 539.0954             | 9.5582             | `download <http://download.deeplite.ai/zoo/models/unet_scse_resnet18-voc-2cls-0_688-79087739621c42c1.pth>`_|
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------------+
| 4 | fcn32                     | 0.713                                                                                                                                | 519.382   | 136.142         | 136.152            | 858.2010             | 24.9806            | `download <http://download.deeplite.ai/zoo/models/fcn32-voc-20_713-b745bd7e373e31d1.pth>`_                 |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------------+
| 5 | deeplab_mobilenet         | 0.571                                                                                                                                | 29.0976   | 26.4870         | 5.8161             | 1134.6057            | 8.2221             | `download <http://download.deeplite.ai/zoo/models/deeplab-mobilenet-voc-20_593-94ac51da679409d6.pth>`_     |
+---+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------------+

.. _zoo_benchmark_results_mnist:

Models on MNIST dataset
-----------------------

+---+---------------------------+---------+----------+-----------------+--------------------+----------------------+--------------------+---------------------------------------------------------------------------------------+
| # | Architecture (model_name) | Top1 (%)| Size (MB)| MACs (Millions) | #Params (Millions) | Memory Footprint(MB) | Execution Time(ms) | Pretrained Weights                                                                    |
+---+---------------------------+---------+----------+-----------------+--------------------+----------------------+--------------------+---------------------------------------------------------------------------------------+
| 1 | lenet5                    | 99.1199 | 0.1695   | 0.2930          | 0.0444             | 0.1904               | 0.4110             | `download <http://download.deeplite.ai/zoo/models/lenet-mnist-e5e2d99e08460491.pth>`_ |
+---+---------------------------+---------+----------+-----------------+--------------------+----------------------+--------------------+---------------------------------------------------------------------------------------+
| 2 | mlp2                      | 97.8046 | 0.4512   | 0.1211          | 0.1183             | 0.4572               | 0.1236             | `download <http://download.deeplite.ai/zoo/models/mlp2-mnist-cd7538f979ca4d0e.pth>`_  |
+---+---------------------------+---------+----------+-----------------+--------------------+----------------------+--------------------+---------------------------------------------------------------------------------------+
| 3 | mlp4                      | 97.8145 | 0.5772   | 0.1549          | 0.1513             | 0.5861               | 0.2356             | `download <http://download.deeplite.ai/zoo/models/mlp4-mnist-c6614ff040df60a4.pth>`_  |
+---+---------------------------+---------+----------+-----------------+--------------------+----------------------+--------------------+---------------------------------------------------------------------------------------+
| 4 | mlp8                      | 96.6970 | 0.8291   | 0.2226          | 0.2174             | 0.8439               | 0.3719             | `download <http://download.deeplite.ai/zoo/models/mlp8-mnist-de6f135822553043.pth>`_  |
+---+---------------------------+---------+----------+-----------------+--------------------+----------------------+--------------------+---------------------------------------------------------------------------------------+

.. _zoo_benchmark_results_cifar100:

Models on CIFAR100 dataset
--------------------------

+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------+
| #  | Architecture (model_name) | Top1 (%) | Size (MB) | MACs (Billions) | #Params (Millions) | Memory Footprint(MB) | Execution Time(ms) | Pretrained Weights                                                                                   |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------+
| 1  | resnet18                  | 76.8295  | 42.8014   | 0.5567          | 11.2201            | 48.4389              | 1.5781             | `download <http://download.deeplite.ai/zoo/models/resnet18-cifar100-86b0c368c511bd57.pth>`_          |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------+
| 2  | resnet50                  | 78.0657  | 90.4284   | 1.3049          | 23.7053            | 123.5033             | 3.9926             | `download <http://download.deeplite.ai/zoo/models/resnet50-cifar100-d03f14e3031410de.pth>`_          |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------+
| 3  | vgg19                     | 72.3794  | 76.6246   | 0.3995          | 20.0867            | 80.2270              | 1.4238             | `download <http://download.deeplite.ai/zoo/models/vgg19-cifar100-6d791de492a133b6.pth>`_             |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------+
| 4  | densenet121               | 78.4612  | 26.8881   | 0.8982          | 7.0485             | 66.1506              | 10.7240            | `download <http://download.deeplite.ai/zoo/models/densenet121-cifar100-7e4ec64b17b04532.pth>`_       |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------+
| 5  | googlenet                 | 79.3513  | 23.8743   | 1.5341          | 6.2585             | 64.5977              | 5.7186             | `download <http://download.deeplite.ai/zoo/models/googlenet-cifar100-15f970a22f56433f.pth>`_         |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------+
| 6  | inception_v4              | 74.7923  | 157.5337  | 7.528           | 41.2965            | 295.4964             | 12.7311            | `download <http://download.deeplite.ai/zoo/models/inceptionv4-cifar100-ad655dfc5fe5b02f.pth>`_       |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------+
| 7  | mobilenet_v1              | 66.8414  | 12.6246   | 0.0473          | 3.3095             | 16.6215              | 1.8147             | `download <http://download.deeplite.ai/zoo/models/mobilenetv1-cifar100-4690c1a2246529eb.pth>`_       |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------+
| 8  | mobilenet_v2              | 73.0815  | 9.2019    | 0.0947          | 2.4122             | 22.8999              | 3.8950             | `download <http://download.deeplite.ai/zoo/models/mobilenetv2-cifar100-a7ba34049d626cf4.pth>`_       |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------+
| 9  | pre_act_resnet18          | 76.5229  | 42.7907   | 0.5566          | 11.2173            | 48.1781              | 1.4383             | `download <http://download.deeplite.ai/zoo/models/pre_act_resnet18-cifar100-1c4d1dc76ee9c6f6.pth>`_  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------+
| 10 | resnext29_2x64d           | 79.9150  | 35.1754   | 1.4167          | 9.2210             | 67.6879              | 2.4351             | `download <http://download.deeplite.ai/zoo/models/resnext29_2x64d-cifar100-f6ba33baf30048d1.pth>`_   |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------+
| 11 | shufflenet_v2_1_0         | 69.9169  | 5.1731    | 0.0462          | 1.356              | 12.3419              | 4.6424             | `download <http://download.deeplite.ai/zoo/models/shufflenet_v2_l.0-cifar100-16ae6f50f5adecad.pth>`_ |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+------------------------------------------------------------------------------------------------------+

.. _zoo_benchmark_results_vww:

Models on VWW dataset
---------------------

+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+-------------------------------------------------------------------------------------------+
| # | Architecture (model_name) | Top1 (%) | Size (MB) | MACs (Billions) | #Params (Millions) | Memory Footprint(MB) | Execution Time(ms) | Pretrained Weights                                                                        |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+-------------------------------------------------------------------------------------------+
| 1 | resnet18                  | 93.5496  | 42.6389   | 1.8217          | 11.1775            | 74.6057              | 1.6008             | `download <http://download.deeplite.ai/zoo/models/resnet18-vww-7f02ab4b50481ab7.pth>`_    |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+-------------------------------------------------------------------------------------------+
| 2 | resnet50                  | 94.3675  | 89.6917   | 4.1199          | 23.5121            | 233.5413             | 4.5085             | `download <http://download.deeplite.ai/zoo/models/resnet50-vww-9d4cb2cb19f8c5d5.pth>`_    |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+-------------------------------------------------------------------------------------------+
| 3 | mobilenet_v1              | 92.4444  | 12.2415   | 0.5829          | 3.2090             | 70.5286              | 1.7777             | `download <http://download.deeplite.ai/zoo/models/mobilenetv1-vww-84f65dc4bc649cd6.pth>`_ |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+-------------------------------------------------------------------------------------------+
| 4 | mobilenet_v3              | 93.3755  | 10.2138   | 0.2723          | 2.6775             | 69.4178              | 5.6989             | `download <http://download.deeplite.ai/zoo/models/mobilenetv3-vww-1d2be1e7d5473081.pth>`_ |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+-------------------------------------------------------------------------------------------+

.. _zoo_benchmark_results_imagenet10:

Models on Imagenet10 dataset
----------------------------

+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+----------------------------------------------------------------------------------------+
| # | Architecture (model_name) | Top1 (%) | Size (MB) | MACs (Billions) | #Params (Millions) | Memory Footprint(MB) | Execution Time(ms) | Pretrained Weights                                                                     |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+----------------------------------------------------------------------------------------+
| 1 | resnet18                  | 93.8294  | 42.6546   | 1.8217          | 11.1816            | 74.6215              | 1.6502             | `download <http://download.deeplite.ai/zoo/models/resnet18-vww-7f02ab4b50481ab7.pth>`_ |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+----------------------------------------------------------------------------------------+
| 2 | mobilenet_v2_0_35         | 81.0492  | 1.5600    | 0.0664          | 0.4089             | 34.9010              | 3.4738             | `download <http://download.deeplite.ai/zoo/models/resnet50-vww-9d4cb2cb19f8c5d5.pth>`_ |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+----------------------------------------------------------------------------------------+

.. _zoo_benchmark_results_imagenet16:

Models on Imagenet16 dataset
----------------------------

+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+----------------------------------------------------------------------------------------+
| # | Architecture (model_name) | Top1 (%) | Size (MB) | MACs (Billions) | #Params (Millions) | Memory Footprint(MB) | Execution Time(ms) | Pretrained Weights                                                                     |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+----------------------------------------------------------------------------------------+
| 1 | resnet18                  | 94.5115  | 42.6663   | 1.8217          | 11.1816            | 74.6332              | 1.6349             | `download <http://download.deeplite.ai/zoo/models/resnet18-vww-7f02ab4b50481ab7.pth>`_ |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+----------------------------------------------------------------------------------------+
| 2 | resnet50                  | 96.8518  | 89.8011   | 4.1199          | 23.5408            | 233.6508             | 4.0444             | `download <http://download.deeplite.ai/zoo/models/resnet50-vww-9d4cb2cb19f8c5d5.pth>`_ |
+---+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+----------------------------------------------------------------------------------------+

.. _zoo_benchmark_results_imagenet:

Models on Imagenet dataset (from torchvision)
---------------------------------------------

+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+
| #  | Architecture (model_name) | Top1 (%) | Size (MB) | MACs (Billions) | #Params (Millions) | Memory Footprint(MB) | Execution Time(ms) | Pretrained Weights |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+
| 1  | resnet18                  | 69.7319  | 44.5919   | 1.8222          | 11.6895            | 76.5664              | 1.6013             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+
| 2  | resnet34                  | 73.2880  | 83.1515   | 3.6756          | 21.7977            | 131.8740             | 2.9303             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+
| 3  | resnet50                  | 76.1001  | 97.4923   | 4.1219          | 25.5570            | 241.3496             | 4.0728             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+
| 4  | resnet101                 | 77.3489  | 169.9416  | 7.8495          | 44.549             | 385.3847             | 7.8066             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+
| 5  | resnet152                 | 78.2836  | 229.6173  | 11.5807         | 60.1928            | 533.4902             | 11.7400            | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+
| 6  | inception_v3              | 69.5109  | 90.9217   | 2.8472          | 27.1613            |  149.3052            | 7.9384             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+
| 7  | densenet121               | 74.4106  | 30.4369   | 2.8826          | 7.9789             | 187.7805             | 10.9740            | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+
| 8  | densenet161               | 77.1120  | 109.4093  | 7.8184          | 28.681             | 393.9603             | 14.4702            | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+
| 9  | densenet169               | 75.5635  | 53.9760   | 3.4184          | 14.149             | 238.9538             | 15.2050            | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+
| 10 | densenet201               | 76.8702  | 76.3471   | 4.3670          | 20.0139            | 307.5974             | 18.8687            | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+
| 11 | alexnet                   | 56.4758  | 233.0812  | 0.7156          | 61.1008            | 237.8486             | 0.4611             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+
| 12 | squeezenet1_0             | 58.0591  | 4.7624    | 0.8300          | 1.2484             | 51.2403              | 1.5843             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+
| 13 | squeezenet1_1             | 58.1438  | 4.7130    | 0.3559          | 1.235              | 32.1729              | 1.7504             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+
| 14 | vgg11                     | 68.9946  | 506.8334  | 7.6301          | 132.8633           | 570.0989             | 0.6626             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+
| 15 | vgg11_bn                  | 70.3433  | 506.8544  | 7.6449          | 132.8688           | 598.4480             | 0.9216             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+
| 16 | vgg13                     | 69.9017  | 507.5373  | 11.3391         | 133.0478           | 607.5527             | 0.7682             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+
| 17 | vgg13_bn                  | 71.5557  | 507.5597  | 11.3636         | 133.0537           | 654.2783             | 1.0535             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+
| 18 | vgg16                     | 71.5605  | 527.7921  | 15.5035         | 138.3575           | 637.7607             | 0.9151             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+
| 19 | vgg16_bn                  | 73.3352  | 527.8243  | 15.5306         | 138.3660           | 689.4726             | 1.2815             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+
| 20 | vgg19                     | 72.3449  | 548.0470  | 19.6679         | 143.6672           | 667.9687             | 1.1326             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+
| 21 | vgg19_bn                  | 74.1900  | 548.0890  | 19.6976         | 143.6782           | 724.6669             | 1.5138             | .                  |
+----+---------------------------+----------+-----------+-----------------+--------------------+----------------------+--------------------+--------------------+

- **Model Size:** Memory consumed by the parameters (weights and biases) of the model
- **MACs:** Summation of Multiply-Add Cumulations (MACs) per single image (batch_size=1)
- **#Parames:** Total number of parameters (trainable and non-trainable) in the model
- **Memory Footprint:** Total memory consumed by the parameters (weights and biases) and activations (per layer) per single image (batch_size=1)
- **Execution Time:** On current device, time required for the forward pass per single image (batch_size=1)

The host machine specifications used to perform the reported benchmarks:

- `NVIDIA TITAN V <https://www.nvidia.com/en-us/titan/titan-v/>`_
- Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz
- 512G SSD HardDrive
- 64G RAM

