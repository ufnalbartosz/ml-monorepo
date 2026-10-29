"""AlexNet implementation on Keras 3 / TensorFlow 2.

Nothing is re-exported here on purpose: importing ``pure_alexnet.oxflower17``
or ``pure_alexnet.dataset`` should not drag in TensorFlow, which takes several
seconds to import.  Reach for the submodule you need:

* :mod:`pure_alexnet.model` - the architecture
* :mod:`pure_alexnet.dataset` - splits and the on-disk cache
* :mod:`pure_alexnet.oxflower17` - download and decode
* :mod:`pure_alexnet.train` - the CLI entrypoint
"""
