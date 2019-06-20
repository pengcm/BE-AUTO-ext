# BE-AUTO-ext
This is the project for our paper "CNN-based bit-depth enhancement by the suppression of false contour and color distortion

file description:
"BE-AUTO-ext": the implementation of paper"photo-realistic image bit-depth enhancement via residual transposed concolutional neural network" and "bit-depth enhancement via convolutional neural network"
""
"BDE-SCD-CNN"the project for our paper "CNN-based bit-depth enhancement by the suppression of false contour and color distortion

experiment environent:
GTX 1050Ti 3G tensorflow-2.0-alpha tensorlayer-2.0.1

acknowledgement: https://github.com/tensorlayer/srgan
how to use:
1.train:
a. you can set yout own parameters in config.py. e.g. your can use your own data set to train by set" config.TRAIN.hr_img_path='your data set path'"
b.the first time you run 'train', the code will automatically download trained 'vgg-19', if it takes too long, you can download it manually from https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs and put it in a folder named 'models'
c. in your command line, type: python train.py 

 
test: in your command line,type:python train.py  --mode evaluate

network model:![maze](https://github.com/pengcm/BE-AUTO-ext/blob/master/model.jpg) 
results:![maze](https://github.com/pengcm/BE-AUTO-ext/blob/master/results/figure6_GT.png)
![maze](https://github.com/pengcm/BE-AUTO-ext/blob/master/results/figure6_ZP.png)
![maze](https://github.com/pengcm/BE-AUTO-ext/blob/master/results/figure6_BE-RTCNN.png)
![maze](https://github.com/pengcm/BE-AUTO-ext/blob/master/results/figure6_ours.png)
![maze](https://github.com/pengcm/BE-AUTO-ext/blob/master/results/figure7_GT.png)
![maze](https://github.com/pengcm/BE-AUTO-ext/blob/master/results/figure7_ZP.png)
![maze](https://github.com/pengcm/BE-AUTO-ext/blob/master/results/figure7_BE-RTCNN.png)
![maze](https://github.com/pengcm/BE-AUTO-ext/blob/master/results/figure7_ours.png)
