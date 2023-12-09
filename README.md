### Packages need to be installed
- pip install numpy
- pip install opencv-python
- pip install keras
- pip3 install --upgrade tensorflow
- pip install pillow

### download FER2013 dataset
- from below link and put in data folder under your project directory
- https://www.kaggle.com/msambare/fer2013

### download Anaconda
- https://www.anaconda.com/download

### Train Emotion detector
- with all face expression images in the FER2013 Dataset
- command --> python TrainEmotionDetector.py

It will take several hours depends on your processor.
after Training , you will find the trained model structure and weights are stored in your project directory.
emotion_model.json
emotion_model.h5

copy these two files create model folder in your project directory and paste it.

### run your emotion detection test file
python TestEmotionDetector.py

###  Graph
Loss
![Machine Learning_TrainTest Plot_loss - Copy](https://github.com/CominS00n/AI-Emotion-detection/assets/79715461/699b8557-f5d9-447e-bf75-45a742ff7e95)

Accuracy
![MachineLearningTrainTestPlotAccuracy](https://github.com/CominS00n/AI-Emotion-detection/assets/79715461/b5eb0a67-c701-40d5-89b9-6a28fbb26b01)

### Example Emotion
![image](https://github.com/CominS00n/AI-Emotion-detection/assets/79715461/03d9f488-2e09-4a76-9f56-bd424a977aae)
![image](https://github.com/CominS00n/AI-Emotion-detection/assets/79715461/02406798-d287-4567-9c54-91ead2d194f2)



