# Facial Emotion Recognition

To complete this task I used YOLOv3 tiny architecture in PyTorch. <br/>
### I used colab to run python scripts using GPU

All files can be found here: https://drive.google.com/open?id=1Y6vVI9YfqX-7RebCCZy9xOHQB3yhaDZy <br/>
Test data: https://drive.google.com/drive/folders/1fQ2xkaZxEPY3ZNjcAE3zFLsegoCfolom <br/>
Test data with bounding boxes: https://drive.google.com/drive/folders/1fU2_YEWLFHmFbJKHzUzeg38TGal0LUJU <br/>
Training data: https://drive.google.com/drive/folders/1-L_OliRDtxtL_7qGjkGTK6_P4ZpraeTJ <br/>

Main functions and architectures were taken from https://github.com/eriklindernoren/PyTorch-YOLOv3

test.py, train.py and detect.py were simplified and decreased in the content from the github repo defined above and adapted to our task.
Furthermore, slight changes in utils/datasets.py have been made. Data have been pre-processed to meet our requirements. You can find training data and its structure from the link above.

All files have been compiled with PyTorch 1.1.0, Torchvision 0.3.0, Pillow 6.1.0.<br/>
Run $sudo pip3 install -r requirements.txt<br/>
In order to meet all requirements<br/>

## What each .py file do:<br/>
1)train.py -> trains the network.<br/>
2)test.py -> is used to do testing on a validation set. <br/>
3)detect.py -> is used to do predictions on a test set<br/>

## How to train the network?<br/>
1) First of all, please download all the files from the google drive by the first link on the top of the Readme <br/>
2) Run $cd yolo_pytorch <br/>
3) Run $python train.py [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--model_def MODEL_DEF] [--data_config DATA_CONFIG]<br/>
4) train.py will save weights in .pth file with the format "epoch_" + str(epoch) +".pth", there will also be .txt file with corresponding loss for each .pth file, in "epoch_" + str(epoch) + "_loss.txt" format <br/>


## How to detect pictures in test set?<br/>
1) As I mentioned above, please download all the files from the google drive by the first link on the top of the Readme <br/>
2) Run $cd yolo_pytorch <br/>
3) Run $python detect.py [--image_folder IMAGE_FOLDER] [--model_def MODEL_DEF] [--weights_path WEIGHTS_PATH] [--class_path CLASS_PATH] [--conf_thres CONF_THRES] [--nms_thres NMS_THRES] [--batch_size BATCH_SIZE] [--img_size IMG_SIZE] <br/>
Note: in [--weights_path WEIGHTS_PATH], please write the recent path of recent .pth file. If not, network trained with 60 epochs (small.pth) will be used<br/>
4) It will save picture with a bounding boxes in yolo_pytorch/data/outcome/<br/>

![alt text](https://github.com/MrPositron/face_emotion_detection/blob/master/young-beautiful-chinese-woman-talking-260nw-1505387699.png) <br/>
Face detection and emotion classification on an image from the test data <br/>

Feel free to contact me at 35646knk {at} gmail {dot} com
