# HyperIQA

This is the source code for the CVPR'20 paper "Blindly Assess Image Quality in the Wild Guided by A Self-Adaptive Hyper Network".

## Dependencies

- Python 3.6+
- PyTorch 0.4+
- TorchVision
- scipy

(optional for loading specific IQA Datasets)
- csv (KonIQ-10k Dataset)
- openpyxl (BID Dataset)

## Training & Testing

Training and testing our model on LIVE Challenge Dataset.

```
python train_test_IQA.py
```

Available options:
* `--dataset`: Training and testing dataset, support datasets: livec | koniq-10k | bid | live | csiq | tid2013.
* `--train_patch_num`: Sampled image patch number from each training image.
* `--test_patch_num`: Sampled image patch number from each testing image.
* `--batch_size`: Batch size.

## Citation
If you find this work useful for your research, please cite our paper:
```
@InProceedings{Su_2020_CVPR,
author = {Su, Shaolin and Yan, Qingsen and Zhu, Yu and Zhang, Cheng and Ge, Xin and Sun, Jinqiu and Zhang, Yanning},
title = {Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive Hyper Network},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```



