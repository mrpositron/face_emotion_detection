# Facial Emotion Recognition

To complete this task I used YOLOv3 tiny architecture in PyTorch. <br/>

All files can be found here: https://drive.google.com/open?id=1Y6vVI9YfqX-7RebCCZy9xOHQB3yhaDZy <br/>
Test data: https://drive.google.com/drive/folders/1fQ2xkaZxEPY3ZNjcAE3zFLsegoCfolom <br/>
Test data with bounding boxes: https://drive.google.com/drive/folders/1fU2_YEWLFHmFbJKHzUzeg38TGal0LUJU <br/>
Training data: https://drive.google.com/drive/folders/1-L_OliRDtxtL_7qGjkGTK6_P4ZpraeTJ <br/>
Note: Bounding Boxes predictions have been made on a network that have been trained with only 20 epochs. I am training it now on 100 epochs, and when it will finish I will update bounding boxes.<br/>

Main functions and architectures were taken from https://github.com/eriklindernoren/PyTorch-YOLOv3

test.py, train.py and detect.py were simplified and decreased in the content from the github repo defined above and adapted to our task.
Furthermore, slight changes in utils/datasets.py have been made.

Data have been pre-processed to meet our requirements.
You can find training data and its structure from the link above.

All files have been compiled with PyTorch 1.1.0, Torchvision 0.3.0, Pillow 6.1.0.<br/>
Run $sudo pip3 install -r requirements.txt<br/>
In order to meet all requirements<br/>

What each .py file do:<br/>
1)train.py -> trains the network for about 100 epochs.<br/>
2)test.py -> is used to do testing on a validation set. <br/>
3)detect.py -> is used to do predictions on a test set<br/>

How to train the network?<br/>
1) First of all, please download all the files from the google drive by the first link on the top of the Readme <br/>
2) Run $cd yolo_pytorch <br/>
3) Run $python train.py <br/>
4) It will save weights in .pth file<br/>
5) There will be several .pth files, just choose the recent now. Training is still going, so I am not sure which .pth file will have minimal loss.

How to detect pictures in test set?<br/>
1) As I mentioned above, lease download all the files from the google drive by the first link on the top of the Readme <br/>
2) Run $cd yolo_pytorch <br/>
3) In detect.py specify which weight file (.pth) you want to load.<br/>
4) Run $python detect.py <br/>
5) It will save picture with a bounding boxes in yolo_pytorch/data/outcome/<br/>

Feel free to contact me at 35646knk {at} gmail {dot} com
