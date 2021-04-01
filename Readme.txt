Compatiblity:
   -name of new env is tensorflow
   -all commands should be run in keras-frcnn folder
1- Create new env name (tensorflow) with Python 3.7 | conda create --name tensorflow python=3.7
	to acitvate new environment		    | conda activate tensorflow 
2- Tensorflow 1.13.1				    | pip install tensorflow==1.13.1
3- Keras 2.0.3					    | pip install keras==2.0.3
4- Numpy					    | pip install numpy
5- Pandas					    | pip install numpy	
6- Hdf5						    | pio install Hdf5

Extract Keras-frcnn.rar
-go to Keras-frcnn

TRAIN: (model is already trained you just need to test)
-python train_frcnn.py -o simple -p annotate.txt (you must be in keras-frcnn folder)
OR
otherwise:
-cd keras_frcnn
-python train_frcnn.py -o simple -p annotate.txt

TEST:
copy images you want to test in test_images folder
-python test_frcnn.py -p test_images

RESULT:
Now goto Keras-frcnn ---> Results_imgs 

For better understanding goto:
https://www.analyticsvidhya.com/blog/2018/11/implementation-faster-r-cnn-python-object-detection/