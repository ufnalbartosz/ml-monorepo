import loader
from plotting import plot_images
from loader import labels
import matplotlib.pyplot as plt

loader.maybe_download_and_extract()

images_test, cls_test, labels_test = loader.load_test_data()

images = images_test[0:9]
cls_true = cls_test[0:9]

plot_images(images=images, cls_true=cls_true, class_names=labels)

plt.show()
