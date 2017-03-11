Kaggle: Dogs vs Cats Kernel Redux Edition
------------------------------------------------------------

This repository consists a solution for the [Cats vs Dogs classification challenge][1] on Kaggle. 
This solution scored a logloss of 0.04973 (#60 / 1314 teams). Top score was 0.03303 (#1).

This repository also contains links to my best and second best CNN models below.

Features
-------------
- Best model (download Keras model [here][2]) achieves accuracy of **99.6%** on unseen data.  
- *Ensemble* of the best model with the second best (download Keras model [here][3]) achieves **99.8%** accuracy on unseen data. Predictions of the two models are simply averaged. 
- The performance is a result of finetuning pretrained models (Resnet-50 & Inception v3 respectively).
- Used several data augmentation techniques -- channel jittering, blurs, brightness-contrast shifts, rotation, zooms,  horizontal-vertical shifts and flips.

What worked
-------------------
- Using ELU activations or BatchNorm + PReLu
- Using Dropout on Dense Layers with Data Augmentation
- Using Pretrained models works like magic! :pray:
- Training small models (10 layers max) from scratch for 25000 images plateaus at double the logloss than when using pretrained models, in my case.  
- Using larger image sizes improves accuracy till 350 x 350
- Ensembling different models gives a very good accuracy boost. Higher the difference (different image sizes, different architectures etc.), better is the advantage of the ensemble.
- Using the trained model to detect mistakes in the training and validation data! 

What didn't turn out as expected
-------------------
- Going beyond the plateau logloss when training from scratch was almost impossible; probably because the lower layers did not learn good enough representations with limited data.
- Trying to finetune last conv block of the pretrained models leads to overfitting in my case. 
- Using Dropout in conv layers leads to poor accuracy
- Training even small models without BatchNorm and only using plain ReLU activations takes a huge amount of time. 
- Mean subtraction + Dividing by Standard Deviation speeds up training 

### Using the trained models
Download the trained models: <br/>
[ResNet50 finetuned (best model)][2] (Input shape: **350 x 350** x 3) <br/>
[InceptionV3 finetuned (second best)][3] (Input shape: **299 x 299** x 3)

```python
from keras.models import load_model
model = load_model('bestvalyet.h5')  # weights path.
```
Both the models use Tensorflow dimension ordering.


  [1]: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition
  [2]: https://drive.google.com/open?id=0B9Hz5dudRW34aldhTlA0RDBjeXc
  [3]: https://drive.google.com/open?id=0B9Hz5dudRW34MVZfdGhxVE5lXzQ
