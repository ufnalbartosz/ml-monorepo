"""CNNs for a 20-class subset of CIFAR-100, on Keras 3 / TensorFlow 2.

Nothing is re-exported here on purpose: importing ``project_cnn.loader`` or
``project_cnn.prepare_dataset`` should not drag in TensorFlow.  Reach for the
submodule you need:

* :mod:`project_cnn.dual_path`, :mod:`project_cnn.inception`,
  :mod:`project_cnn.alexnet_model` - the three architectures
* :mod:`project_cnn.loader`, :mod:`project_cnn.prepare_dataset` - the data
* :mod:`project_cnn.tools` - pre-processing and model introspection
* :mod:`project_cnn.evaluate`, :mod:`project_cnn.plot` - measuring and plotting
* :mod:`project_cnn.main` - the CLI entrypoint
"""
