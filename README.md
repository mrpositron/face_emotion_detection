# Facial Emotion Recognition

To complete this task I used YOLOv3 tiny architecture in PyTorch. <br/>

All the file can be found here: https://drive.google.com/open?id=1Y6vVI9YfqX-7RebCCZy9xOHQB3yhaDZy <br/>
Test data: https://drive.google.com/drive/folders/1fQ2xkaZxEPY3ZNjcAE3zFLsegoCfolom <br/>
Test data with bounding boxes: https://drive.google.com/drive/folders/1fU2_YEWLFHmFbJKHzUzeg38TGal0LUJU <br/>
Training data: https://drive.google.com/drive/folders/1-L_OliRDtxtL_7qGjkGTK6_P4ZpraeTJ <br/>

Main functions and architectures were taken from https://github.com/eriklindernoren/PyTorch-YOLOv3

test.py, train.py and detect.py were simplified and decreased in the content from the github repo defined above and adapted to our task.
Furthermore, slight changes in utils/datasets.py have been made.

Data have been pre-processed to meet our requirements.
You can find training data and its structure from the link above.

All files have been compiled with PyTorch 1.1.0, Torchvision 0.3.0, Pillow 6.1.0.<br/>
Run $sudo pip3 install -r requirements.txt<br/>
In order to meet all requirements<br/>

What each .py file do:<br/>
1)train.py -> trains the network for about 100 epochs. every 20 epochs saves the network weights.<br/>
2)test.py -> is used to do testing on a validation set. <br/>
3)detect.py -> is used to do predictions on a test set<br/>

How to train the network?<br/>
Run $python train.py <br/>
It will save weights in small.pth<br/>

How to detect pictures in test set?<br/>
Run $python test.py<br/>
It will automitcally load weights (small.pth file) that have been saved during training.<br/>
Furthermore, it will save picture with a bounding boxes in /data/outcome/<br/>
