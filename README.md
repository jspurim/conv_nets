# Convolutional neural networks

Just a bunch of CNNs scripts and utils.

Some of this stuff is based on the code from [this tutorial](https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/).

To train a model just run

```
$ python network_trainer.py -d <datset_folder> -m <model_output_file> <network>
```

Parameters for a specific network need to be supplied after the network in the command line.

To test a model run

```
$ python network_test.py -m <model> -i <image> -l <label_for_0 label_for_1 ...>
```
