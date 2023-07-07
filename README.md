# PiLLE: Physics-inspired Contrastive Learning for Low-light Enhancement


****

## Experiment

PyTorch implementation of PiLLE

### Requirements

- Python 3.8 
- PyTorch 1.4.0
- opencv
- torchvision 
- numpy 
- pillow 
- scikit-learn 
- tqdm 
- matplotlib 
- visdom 

SCL-LLE does not need special configurations. Just basic environment.

### Folder structure

The following shows the basic folder structure.
```python
├── datasets
│   ├── data
│   │   ├── cityscapes
│   │   └── Contrast
|   ├── test_data
│   ├── cityscapes.py
|   └── util.py
├── lowlight_test.py # low-light image enhancement testing code
├── train.py # training code
├── lowlight_model.py
├── Myloss.py
├── CR.py
├── curves.py
├── checkpoints
│   ├── PiLLE.pth #  A pre-trained PiLLE model
```

### Test

- cd PiLLE


```
python lowlight_test.py
```

The script will process the images in the sub-folders of "test_data" folder and make a new folder "result" in the "datasets". You can find the enhanced images in the "result" folder.

### Train

1. cd PiLLE
2. download the [Cityscapes](https://www.cityscapes-dataset.com/) dataset
3. download the cityscapes training data <a href="https://drive.google.com/file/d/1FzYwO-VRw42vTPFNMvR28SnVWpIVhtmU/view?usp=sharing">google drive</a> and contrast training data <a href="https://drive.google.com/file/d/1A2VWyQ9xRXClnggz1vI-7WVD8QEdKJQX/view?usp=sharing">google drive</a> 
4. unzip and put the downloaded "train" folder and "Contrast" folder to "datasets/data/cityscapes/leftImg8bit" folder and "datasets/data" folder


```
python train.py
```


## Contact
If you have any question, please contact xuzhengyan@nuaa.edu.cn
