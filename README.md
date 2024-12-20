# JCDM
Joint Conditional Diffusion Model for Image Restoration with Mixed Degradations

![method](https://github.com/mengyu212/JCDM/blob/main/fig5.jpg)

## Train and test
### dataset

### training
For the first stage training, modify the path of the dataset in config/diffusion.json.
```
python train_stage1.py
```
The trained models will be placed in the experiments/checkpoint folder.
### testing
For the first stage testing, modify the path of the dataset and the pretrained model in config/infer.json.
```
python test_stage1.py
```
