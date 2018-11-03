# FaceNet 
This is an implementation of Google's FaceNet  network in tensorflow. 

# Data Prep
I use dlib's stock face extractor as this will be deployed on a raspberry pi. You can
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

# Training Options
There is a classification based model and a triplet model. Each requires a different data format -- json for triple csv for
classification.