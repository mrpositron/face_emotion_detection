# Facial Emotion Recognition

To complete this task I used YOLOv3 tiny architecture in PyTorch. 

Main functions and architectures were taken from https://github.com/eriklindernoren/PyTorch-YOLOv3

test.py, train.py and detect.py were simplified and decreased in the content from the github repo defined above and adapted to our task.
Furthermore, slight changes in utils/datasets.py have been made.

Data have been pre-processed to meet our requirements.
You can find data file by opening this link:

All files have been compiled with PyTorch 1.1.0, Torchvision 0.3.0, Pillow 6.1.0.<br/>
Run $sudo pip3 install -r requirements.txt<br/>
In order to meet all requirements<br/>

What each .py do:<br/>
1)train.py -> trains the network for about 100 epochs. every 20 epochs saves the network weights.&nbsp<br/>
2)test.py -> is used to do testing on a validation set. &nbsp<br/>
3)detect.py -> is used to do predictions on a test set<br/>

