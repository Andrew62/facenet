# FaceNet 
This is an implementation of Google's FaceNet 
network in tensorflow. 

# Data Prep
I only use dlib's stock face extractor as this will be deployed on a raspberry pi. You can
likely achieve better face extraction using other methods. Be sure to orgranize 
your data into 
    
    name1
        1.jpg
        2.jpg
    name2
        1.jpg
        2.jpg
        
The naming convetion doesn't matter so much as the folder structure and this 
will decode .jpg, .png, and .gif.

```bash
python  data_prep/extract_faces.py -i /path/to/source/dir -o /path/to/output/dir
```

Then collect faces into a json file

```bash
python  data_prep/extract_faces.py -i /path/to/source/dir -o /path/to/output.json --min_examples 2
```

# Training
Model training starts from `inception_resnet_v2.py` trained on ImageNet.
The base model can be [downloaded from here on the TF Slim page](http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz).
From there, we set the number of classes to the embedding size, partially load
the checkpoint excluding `InceptionResnetV2/Logits`, `InceptionResnetV2/AuxLogits`, and
optimizer variables. In this case that's `RMSProp`. 

No enqueue ops were used since it doesn't take much time for the data to load 
(~0.01 seconds per batch of 64 images) and it's also easier to convert for inference
tasks.

The key to fast convergence as noted in the original paper is hard triplet mining where
you attempt to find an anchor A, a positive example P, and a negative example N such that
L2(A-P)^2 > L2(A-N)^2. **Important** implementation note when using 
Tensorflow's `tf.nn.ls_normalize(logits, 1)` is to normalize along the 1st dim. As also noted
in the paper, it is inefficient to find optimal samples in the dataset so we 
sample within subsets. This implementation samples 100 identities with a max of 25 images
per identity by default then evaluates all combinations for each to generate triplets. The proceedure 
is to process all images through the network, generate triplets based off the current network state
then do a complete pass with the ne triplets. 

After data prep you can run:

```bash
python train.py -i /path/to/faces.json -c /path/to/checkpoint/dir/ -p /path/to/inception_resnet_v2.ckpt
```

# Command Line Inference

Run inference with a model using the following script. This script takes advantage of
embeddings generated at the end of the training script to perform pairwise comparisons

```bash
python inference.py -i /path/to/image/1.jpg /path/to/image/2.jpg -c /path/to/checkpoint
```
