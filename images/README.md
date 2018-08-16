# Images
Download [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) and unpack into the images directory using the [torchvision datasets format](https://pytorch.org/docs/stable/torchvision/datasets.html#datasetfolder).

Image files will be arranged the following way. Directory "1" represents the
class by the name "1" and is how we identify which class of flower the image is.
These classes then map to `..\cat_to_name.json`.
```
images/test/1/12345.jpg
images/train/1/23456.jpg
images/valid/1/34567.jpg
```

