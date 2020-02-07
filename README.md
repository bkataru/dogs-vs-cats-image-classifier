# Dogs vs cats image classifier

A CNN to classify images of dogs and cats built using three architectures - a pretrained VGG16 network, an untrained VGG16 network,
and a custom model built from scratch by me.

Trained and tested on 16 gigs of RAM, i7-8750H, GTX 1060 all at stock settings.

Check `requirements.txt` and `packages_to_install.txt` for module information.

## To-do:
- Print labels properly for untrained and custom model.
- Get untrained and custom model working.

## Quick start

### To train: 
You can change training parameters such as Steps Per Epoch, Batch Size (default 32), Epochs, Validation Steps in the python file `pretrained_vgg16_train.py`
```shell
python pretrained_vgg16_train.py
```

### To test 
You can change the testing image used inside the python file `pretrained_vgg16_test.py`
```shell
python pretrained_vgg16_test.py
```
