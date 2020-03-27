# BE-AUTO-ext
This is the project for our paper "CNN-based bit-depth enhancement by the suppression of false contour and color distortion

file description:
"BE-AUTO-ext": the implementation of our paper "CNN-based bit-depth enhancement by the suppression of false contour and color distortion". 

experiment environent:
GTX 1050Ti 3G tensorflow-2.0-alpha tensorlayer-2.0.1

acknowledgement: https://github.com/tensorlayer/srgan

how to use:

1.train:
a. you can set yout own parameters in config.py. e.g. your can use your own data set to train by set" config.TRAIN.hr_img_path='path of your data set'"
b.the first time you run 'train', the code will automatically download trained 'vgg-19', if it takes too long, you can download it manually from https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs and put it in a folder named 'models'
c. in your command line, type: python train.py 

 
test: in your command line,type:python train.py  --mode evaluate

Possible bugs under windows:

a. "NameError: name 'glob' is not defined"  HOW TO FIX: put 'import glob' on a new line, instead of on the line 1 with other packages. The same problem may occur for "os" package. 

b."FileNotFoundError: file checkpoint/g_srgan.h5 dosen't exist"   HOW TO FIX: check the path of your checkpoint file. You need to confirm whether to use absolute or relative paths.

c."InvalidArgumentError: Incompatible shapes: [0] vs. [3]"   HOW TO FIX: check your dataset to see wether every image meets data integrity.

network model:![maze](https://github.com/pengcm/BE-AUTO-ext/blob/master/model.jpg) 
results:![maze](https://github.com/pengcm/BE-AUTO-ext/blob/master/results/figure5_GT.png)
![maze](https://github.com/pengcm/BE-AUTO-ext/blob/master/results/figure5_ZP.png)
![maze](https://github.com/pengcm/BE-AUTO-ext/blob/master/results/figure5_BE-RTCNN.png)
![maze](https://github.com/pengcm/BE-AUTO-ext/blob/master/results/figure5_ours.png)
![maze](https://github.com/pengcm/BE-AUTO-ext/blob/master/results/figure6_GT.png)
![maze](https://github.com/pengcm/BE-AUTO-ext/blob/master/results/figure6_ZP.png)
![maze](https://github.com/pengcm/BE-AUTO-ext/blob/master/results/figure6_BE-RTCNN.png)
![maze](https://github.com/pengcm/BE-AUTO-ext/blob/master/results/figure6_ours.png)
