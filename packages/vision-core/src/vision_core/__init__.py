"""Shared building blocks for the image-classification packages in this workspace.

`pure-alexnet` and `project-cnn` train different models on different data-sets,
so they stay separate packages.  What they had in common - fetching an archive,
caching a prepared data-set as a pickle, the Keras training loop, checkpoint
callbacks, accuracy - lived twice, once in each, and drifted.  It lives here.

Nothing is re-exported at package level: `vision_core.cache`,
`vision_core.archives`, `vision_core.images` and `vision_core.labels` are pure
Python and numpy, and importing them should not pull in TensorFlow.  Only
`vision_core.training` and `vision_core.metrics` need Keras.
"""
