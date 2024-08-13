# NAIC
This repository contains the reference code for the paper "Exploring Patterns and Semantics in Non-Autoregressive Image Captioning" .

# Requirements
- Python 3.7
- Pytorch 1.12
- Torchvision 0.13
- timm 0.6.11
- numpy
- tqdm
- [clip](https://github.com/openai/CLIP)
- [coco_caption](https://github.com/tylin/coco-caption)

# Preparation
## 1. Data preparation
The necessary files in training and evaluation are saved in `mscoco` folder, which is organized as follows:
```python
mscoco/
|--feature/
  |--coco2014/
    |--train2014/
    |--value2014/
    |--test2014/
|--misc/
|--sent/
  |--kd/
  |--label_selection/
|--txt/
```
Download annotation files from [GoogleDrive](https://drive.google.com/drive/folders/1dLMx-NBum0GmtSJlZyp19nyVqJDcVwCh?usp=drive_link) and the image Swin feature from [MSCOCO 2014](https://drive.google.com/drive/folders/1D8J95L4Bhxsn3xOOHWNHd-C_tASlPdHh?usp=drive_link) and put them into `mscoco/`.



# Training
You can download the pre-trained model and configs from [GoogleDrive](https://drive.google.com/drive/folders/1rNoRgTOCmh8qlLg0V8zTcDvLzEtn1jfJ?usp=drive_link) and put it into corresponding folders in `experiments/`.
```python
bash experiments/NAIC/train.sh
```

# Evaluation
The metrics: clip score, cider, and rouge are comparable in label selection, all the pretrained weights are provided.
```python
bash experiments/eval.sh
```
| Model       | BLEU-1      | BLEU-2      | BLEU-3      | BLEU-4      | METEOR      | ROUGE-L     | CIDEr       |     
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Ours-KD     | 79.9        | 63.8        | 49.1        | 37.3        | 28.2        | 58.1        | 123.7       |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Ours-COCO   | 80.0        | 63.8        | 49.1        | 37.2        | 28.3        | 58.2        | 123.6       |


# Acknowledgements
Thanks the original [PureT](https://github.com/232525/PureT), [JDAI-CV/image-captioning](https://github.com/JDAI-CV/image-captioning), [EENAIC](https://github.com/Liu-Yuanqiu/EENAIC), and [CLIP](https://github.com/openai/CLIP).
