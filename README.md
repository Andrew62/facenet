# FaceNet 
This is an implementation of Google's FaceNet 
network in tensorflow. At this point it is set up to train
 but with one difference: I'm feeding in positive examples 
 instead of finding *hard* positives within a minibatch.
 
 `Dockerfile.tf-gpu` will build the intended environment 
 if you have `nvidia-docker`. This can be paired with 
 [sentry](https://github.com/Andrew62/sentry) for face 
 identification and extraction. 