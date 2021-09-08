.. _deeper:

**************************
Going Deeper with Neutrino
**************************

By default, Neutrino is wired for optimizing a classification task that has a fairly simple setup. This imposes tight constraints
on the assumed structure of how tensors flow from the data loader, to the model, to the loss function and to the evaluation.

More complex and custom tasks can be supported by Neutrino by following some additional steps. The three main pieces are how to extract
what is coming out of the data loader, the loss function and the evaluation function. Finally, we need some information about the optimizer 
used for the provided pretrained model in order to make it all work.

- :ref:`forward_pass`
    - :ref:`forward_pass_input_pattern`
    - :ref:`forward_pass_common_inputs`
    - :ref:`forward_pass_interface`
- :ref:`loss_function`
    - :ref:`loss_function_interface`
- :ref:`eval_function`
    - :ref:`eval_function_interface`
- :ref:`optimizer`
- :ref:`scheduler`
- :ref:`deep_wrapup`

.. _forward_pass:

Customize Forward Pass
======================

One of the most important things when using the engine for customized data and models is to tell it how to extract
the tensors from the data loader and trigger forward passes on the model object. We provide a modular interface
that needs to be implemented only on the non-standard cases. Also, the interface can default to a working implementation
if provided with two keywords, ``model_input_pattern`` and ``expecting_common_inputs``.

.. _forward_pass_input_pattern:

input pattern
-------------

By default, we assume that your data loader returns a 2-tuple where the first element goes to the model
and the second element goes to the loss function, we call this the standard (x, y) pattern. If the model
signature of its callable expect more than simply x, this can be made available by providing
a tuple of integers and '_' as value to ``model_input_pattern``. The integers represent the order on which to pipe
the element of data loader tuple to the model and the '_' are placeholders to ignore.
It is easier to understand with a few examples:

.. code-block:: python

    # default classification
    model_input_pattern = (0, '_')
    x, y = next(dataloader)
    logits = model(x)

    # example 1
    model_input_pattern = (1, 0, '_', 2)
    a, b, c, d = next(dataloader)
    out = model(b, a, d)

    # example 2
    model_input_pattern = (0, '_', 1, '_', '_')
    a, b, c, d, e = next(dataloader)
    out = model(a, c)

.. _forward_pass_common_inputs:

common inputs
-------------

Another assumption is that the value of ``expecting_common_input`` is True. This translates to the fact that we expect
each element of the data loader tuple to be standard. Standard here means that they are either tensor or
common container types, list, tuple or dict, of tensors. This grants some abilities like being able to automatically
infer the shapes of the input tensors or their numeric types for example. If the output of your data loader is not
standard, then it is required to implement some methods of the interface.

.. _forward_pass_interface:

interface
---------

.. code-block:: python

    class ForwardPass(ABC):
        def __init__(self, model_input_pattern=None, expecting_common_inputs=True):
            """ init """

        def model_call(self, model, x, device):
            """
            Call the model with 'x' extracted from a loader's batch and on device 'device'.
            'x' is literally := x = forward_pass.extract_model_inputs(next(dataloader))

            Default implementation is provided if the ForwardPass is instantiated expecting common inputs.
            """

        def create_random_model_inputs(self, batch_size):
            """
            Create a compatible random input of corresponding 'batch_size'. Compatible in the sense that
            `model_call` can run without crashing this return value.

            Default implementation is provided if the ForwardPass is instantiated expecting common inputs.
            """

        def extract_model_inputs(self, batch):
            """
            Extract a compatible input from a loader's 'batch'. Compatible in the sense that
            `model_call` can run without crashing this return value.

            Default implementation is provided if the ForwardPass is instantiated with a pattern.
            """

        def get_model_input_shapes(self):
            """
            Returns a tuple of all input shapes that are fed to the model.

            Default implementation is provided if the ForwardPass is instantiated expecting common inputs.
            """

.. note::

    When subclassing or only using the features ``model_input_pattern`` and ``expecting_common_inputs``
    you have to use the framework specific ``ForwardPass``. An example can be found at the end :ref:`deep_wrapup`.

The following example shows how to implement the ForwardPass in the case you cannot activate both inner default
implementations.

.. code-block:: python

    class ClassificationTorchForwardPass(ForwardPass):
        def __init__(self):
            super().__init__(model_input_pattern=None, expecting_common_inputs=False)

        def model_call(self, model, x, device):
            # this is built on the assumption that you know how to call your model.
            # imagine here that its like 'def forward(self, x, z)'
            x, z = x # this comes from the output of the method `extract_model_inputs`
            if device == Device.GPU:
                x, z = x.cuda(), z.cuda()
            else:
                x, z = x.cpu(), z.cpu()
            return model(x, z)

        def create_random_model_inputs(self, batch_size):
            shapes = self.get_model_input_shapes()
            return torch.rand(batch_size, *shapes[0]), torch.rand(batch_size, *shapes[1])

        def extract_model_inputs(self, batch):
            x, y, z = batch
            return x, z

        def get_model_input_shapes(self):
            # imagine your model input data have these shapes
            return (3, 32, 32), (100,)

.. _loss_function:

Customize Loss Function
=======================

The next class that needs an implementation is the ``LossFunction``. This is a straightforward interface that needs to be implemented 
is ``__call__`` which accepts the ``model`` and a ``batch``. ``model`` has exactly the same call signature as the one you have provided to the 
engine and ``batch`` is an element in the iteration over your data loader. There is much freedom as to what can happen there. It simply needs 
to return a ``dict`` of tensors that will be summed or a single tensor to yield the scalar for backprop.

.. _loss_function_interface:

interface
---------

.. code-block:: python

    class LossFunction(ABC):
        def __init__(self, device=Device.CPU):
            self._device = None
            self.to_device(device)

        def to_device(self, device):
            """
            Optionally do something if there is a device switch
            """
            self._device = device

        @property
        def device(self):
            return self._device

        @abstractmethod
        def __call__(self, model, batch):
            raise NotImplementedError

The following example shows how to implement CrossEntropy loss function by this interface.

.. code-block:: python

    class ClassificationLoss(LossFunction):
        def __call__(self, model, batch):
            x, y = batch
            if self.device == Device.GPU:
                x, y = x.cuda(), y.cuda()
            else:
                x, y = x.cpu(), y.cpu()
            out = model(x)
            return {'loss': F.cross_entropy(out, y)}

.. _eval_function:

Customize Evaluation Function
=============================

The last class that needs an implementation is the ``EvaluationFunction``. Only the ``apply`` method needs to
be implemented and there is even more flexibility than for the ``LossFunction``. It receives your ``model``
and your ``loader`` as input and it is expected to return a ``dict`` of metrics you wish to keep track of.

.. important::

    You are free to return multiple evaluation metrics that we are going to report from the evaluation function.
    However, the engine can only listen to one at a time (this is the value that has to be specified in the config as **eval_key**.)

.. _eval_function_interface:

interface
---------

.. code-block:: python

    class TorchEvaluationFunction(EvaluationFunction):
        @abstractmethod
        def _compute_inference(self, model, data_loader, device=Device.CPU, transform=None):
            raise NotImplementedError("Base class call")

The following example shows how to implement top1 accuracy eval function for classification task for PyTorch (TorchFramework).

.. code-block:: python

    from deeplite.torch_profiler.torch_inference import TorchEvaluationFunction
    
    class EvalAccuracy(TorchEvaluationFunction):
        def __init__(self, device='cuda'):
            self.device = device

        def _compute_inference(self, model, data_loader, **kwargs):
            total_acc = 0
            with torch.no_grad():
                for x, y in data_loader:
                    if self.device == 'cuda':
                        x, y = x.cuda(), y.cuda()
                    else:
                        x, y = x.cpu(), y.cpu()
                    out = model(x)

                    out = F.softmax(out, dim=-1)
                    out = torch.argmax(out, dim=1)

                    if out.dim() == 1 and y.dim() == 2 and y.shape[1] == 1:
                        y = y.flatten()

                    acc = torch.mean((out == y).float())
                    total_acc += acc.cpu().item()
                return {'accuracy': 100. * (total_acc / float(len(data_loader)))}

.. _optimizer:

Customize Optimizer
===================

It is important that the optimizer used to train the model is the same as the one we will use internally.
There are two ways to bring your optimizer into the engine:

1. A ``dict`` format enables an optimizer directly importable from the framework library. The ``dict`` needs to have a
   `'name'` key that points to the optimizer class to import and all the rest of the items are key value
   pairs used to instantiate it.
2. Implementing Neutrino's interface ``NativeOptimizerFactory``.

.. code-block:: python

    # If you use SGD with 0.1 learning rate, we would need
    optimizer = {'name': 'SGD', 'lr': 0.1}
    # this allow such a thing to happen:
    #   from torch.optim import SGD
    #   opt = SGD(lr=0.1)

    # Now an implementation of the interface:
    class NativeOptimizerFactory(ABC):
        @abstractmethod
        def make(self, native_model):
            """ Returns a native optimizer object """

    class CustomOptimizerFactory(NativeOptimizerFactory):
        def make(self, native_model):
            from torch.optim import Adam
            return Adam(native_model.parameters(), lr=1e-4, betas=(0.9, 0.9))
    optimizer = CustomOptimizerFactory()

.. _scheduler:

Customize Scheduler
===================

It is also possible to provide a scheduler and is recommended to do so if it was used to train the original model.
The scheduler **has** to be given as a ``dict`` with keys `'factory'` and `'eval_based'`.

* `'factory'` is the factory pattern to bring in the scheduler which follows the same structure as the optimizer.
  There are two ways to bring your scheduler into the engine:

    1. A ``dict`` format enables an scheduler directly importable from the framework library. The ``dict`` needs to have a
       `'name'` key that points to the scheduler class to import and all the rest of the items are key value
       pairs used to instantiate it.
    2. Implementing Neutrino's interface ``NativeSchedulerFactory``.

* `'eval_based'` is a ``bool`` that informs Neutrino this scheduler listens to the evaluation metric (much like
  early stopping does). It defaults to ``False``.

.. code-block:: python

    # If using the pytorch scheduler that reduces the learning rate by some factor at every patience count.
    # Note that this scheduler listens to the evaluation metric (ex.: accuracy) to guide its schedule.
    scheduler = {'factory': {'name': 'ReduceLROnPlateau', 'mode': 'max', 'patience': 10, 'factor': 0.2},
                 'eval_based': True}

    # Now an implementation of the interface:
    class NativeSchedulerFactory(ABC):
        @abstractmethod
        def make(self, native_optimizer):
            """ Returns a native scheduler object """

    class CustomSchedulerFactory(NativeSchedulerFactory):
        def make(self, native_optimizer):
            from torch.optim.lr_scheduler import MultiplicativeLR
            return MultiplicativeLR(native_optimizer, lr_lambda=lambda epoch: 0.95)
    scheduler = {'factory': CustomSchedulerFactory(),
                 'eval_based': False}


.. _deep_wrapup:

Wrapping it up together
=======================

Here is an example of how the call to the engine would be made with all those specifications. Please notice
that we do not show here all the possibilities in the ``ForwardPass`` object. We only use the
``model_input_pattern`` (and by default ``expecting_common_inputs`` is True).

.. code-block:: python

    from deeplite.torch_profiler.torch_data_loader import TorchForwardPass as FP
    forward_pass = FP(model_input_pattern=(0, '_', '_'))
    eval_func = MyEvalFunc()

    config = {
        'deepsearch': args.deepsearch, #(boolean),
        'level': args.level, # int {1, 2}
        'delta': args.delta, #(between 0 to 100),
        'device': args.device, # 'GPU' or 'CPU'
        'use_horovod': args.horovod, #(boolean),
        'full_trainer': {
            'optimizer': {'name': 'SGD', lr: 0.1}, # optimizer in a dict format
            'scheduler': {'factory': MySchedulerFactory(), 'eval_based': isEvalBased}, # scheduler in custom factory format
            'epochs': 100, # int for nb of epochs required
            'eval_freq': 2, # useful if the evaluation takes a lot of time
            'eval_key': 'mykey', # str to take from the dict return by MyEvalFunc
            'eval_split': 'test',
        }
    }
    neutrino = Neutrino(framework=framework,
                        data=data_splits,
                        model=reference_model,
                        config=config,
                        forward_pass=forward_pass,
                        eval_func=eval_func,
                        loss_function_cls=MyLoss,
                        loss_function_kwargs=my_loss_config)
    optimized_model = neutrino.run()


.. warning::
    
    Neutrino tries to do model analysis to help improve metric retention while compressing. Doing so requires
    turning a PyTorch model into a graph using PyTorch's own JIT infrastructure. Therefore, the compression
    capacity of the engine can be dramatically harmed if the model cannot be turned into a PyTorch graph due
    to JIT's limitations. One JIT limitations to watch out for is that every `torch.nn.Module`
    (i.e.: the model and its intermediate layers) should return `Tensor` or `list`/`tuple` of `Tensor`.
    
.. warning::
    
    Neutrino needs to keep copies of the model in order to test different variants. Therefore, the function ``deepcopy`` 
    of standard python needs to be able to return a copy of the model without crashing. In PyTorch a common pitfall that prevents 
    ``deepcopy`` is to assign arbitrary `Tensor` (or something that contains `Tensor`) as an attribute of the `torch.nn.Module`.
    PyTorch supports ``deepcopy`` only for `Parameter` tensors, not arbitrary ones. 

.. important::

    For object detection and segmentation models, the community version displays the results of the optimization process including all the optimized metric values. To obtain the optimized model produced by Deeplite Neutrino, consider upgrading to the production version. Refer :ref:`how to upgrade <feature_comparison>`.

.. important::

    Currently, the multi-GPU support is available only for the Production version of Deeplite Neutrino. Refer, :ref:`how to upgrade <feature_comparison>`.
