# Implementing Face Recognition
This repo has two implementations of face recognition (FR), one based on 
Google's [FaceNet](https://arxiv.org/abs/1503.03832) paper using triplet loss and another using a squeeze layer
seen in [OpenFace](https://github.com/cmusatyalab/openface/) and [FaceNet](https://github.com/davidsandberg/facenet) repos. This work was inspired by multiple
different projects and research. This work in my exploration into learning the inner workings of triplet loss
and a desire to connect the capability to a Raspberry Pi for a DIY FR system. 


# Training Data
Since there are two different types of networks -- triplet loss and a compression layer net -- there are two 
different input data formats. Both models were trained on the [VGG Face Dataset](http://www.robots.ox.ac.uk/~vgg/data/vgg_face/) using
[dlib's](http://dlib.net/) frontal face detector to extract face chips.
 

## Triplet Loss
Data is organized as follows:

    {0: [1.jpg, 2.jpg...], 1: [1.jpg, 2.jpg...], ...}
Each json key indicates a new class and each array holds the values. This facilitates
triplet mining and creation. 

## Squeeze Layer 
Data is organized as a typical classification task where the first
column is the id and second is the image file path for example:
    
    19,/path/to/image/1.jpg
    209,/path/to/image/face3.jpg
    85,/path/to/image/photo.jpg
    ...
    
Note no headers are used. 


# CNN Architecture
Theoretically you're supposed to be able to use any network architecture 
for feature extraction. However, after experimenting with several different models
I've found [InceptionV3](https://arxiv.org/abs/1512.00567) to work the best, and followed their 
training regiment using gradient clipping and RMSProp.

# Training your own
There are two scripts, `train_classifier.py` and `train_triplet.py` that hold
the training params and scripts. Each script has a param object that can be configured but it's easiest
to go with the defaults 

With data in place, creating a training environment is done with nvidia-docker using tensorflow 1.10
    
    nvidia-docker run -it -d -p 8888:8888 -p 6006:6006 --name facenet -v ~/net:/notebooks/net tensorflow/tensorflow:latest-gpu-py3
    
Navigate to the notebook (using `docker logs facenet --tail 25` if you need the access token), open a terminal and 
run `python3 train_classifier.py` or `python train_triplet.py` then open a new terminal and run `tensorboard --logdir=.` to track 
progress.

## Projector
By default, `train_classifier.py` has the ability to visualize embeddings. If you would like to link a sprite, 
see `utils/sprite.py` for tools to generate the sprite jpeg and add the following snippet to `train_classifier.py`
before the final save

```python
    prj_config = projector.ProjectorConfig()
    face_centers_prj = prj_config.embeddings.add()
    face_centers_prj.tensor_name = 'centers:0'  # pulled from the variable within losses.py
    face_centers_prj.metadata_path = projector_metadata
    face_centers_prj.sprite.image_path = projector_sprite
    face_centers_prj.sprite.single_image_dim.extend(thumbnail_size)
    projector.visualize_embeddings(summary_writer, prj_config)
```