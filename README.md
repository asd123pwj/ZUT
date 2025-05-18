# Fair Training with Zero Inputs [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/32676/34831) [[Oral Presentation/Poster]](https://drive.google.com/drive/folders/1hWFJ2n4LnGnnD2r-C6dLyE31-0UJldfw?usp=drive_link)

This repository is the official implementation for semantic segmentation in our AAAI 2025 oral presentation paper *Fair Training with Zero Inputs*.

- This code on semantic segmentation is based on [MMSegmentation 1.2.1](https://github.com/open-mmlab/mmsegmentation/tree/v1.2.1).
- Experiments on image classification are based on [MMPretrain 1.1.1](https://github.com/open-mmlab/mmpretrain/tree/v1.1.1).
- Experiments on clothes changing person re-identification are based on [Simple-CCReID](https://github.com/guxinqian/Simple-CCReID).

## Results

<img src="images\ADE20K.png" alt="Results on ADE20K" style="zoom: 25%;" />

**Note**: This repository is a clean reimplementation based on MMSegmentation after paper acceptance, enabling direct comparison with the original codebase to inspect ZUT's modifications. Sample training results on VAN-b0 are shown in the table below.

| Method | Reported mAcc (%) | Reported mIoU (%) | Sample mAcc (%) | Sample mIoU (%) | Logs/Weight                                                                                          |
| ------ | ----------------- | ----------------- | --------------- | --------------- | ---------------------------------------------------------------------------------------------------- |
| VAN-b0 | 52.56             | 37.87             | 53.33           | 38.10           | [Google Drive](https://drive.google.com/drive/folders/1hWFJ2n4LnGnnD2r-C6dLyE31-0UJldfw?usp=drive_link) |

## Install

1. Following the install steps of [MMSegmentation 1.2.1](https://github.com/open-mmlab/mmsegmentation/tree/v1.2.1).
2. Download [ADE20K](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) dataset, and modify `data_root` in `./configs/_base_/datasets/ade20k.py` to match dataset path.
3. Download pretrained weight for [VAN-{b0, b1, b2, b3}](https://cloud.tsinghua.edu.cn/d/0100f0cea37d41ba8d08/), and put them like `./pretrained/van_b0.pth`.
   - MMSeg will auto download pretrained weights for ResNet-50, Poolformer-s12, and ConvNeXt-tiny.

## Training

```python
# ZUT
python tools/train.py configs/0_ZUT/van_b0.py
python tools/train.py configs/0_ZUT/van_b1.py
python tools/train.py configs/0_ZUT/van_b2.py
python tools/train.py configs/0_ZUT/van_b3.py
python tools/train.py configs/0_ZUT/r50.py
python tools/train.py configs/0_ZUT/poolformer_s12.py
python tools/train.py configs/0_ZUT/convnext_tiny.py

# Baseline
python tools/train.py configs/0_ZUT_baseline/van_b0.py
python tools/train.py configs/0_ZUT_baseline/van_b1.py
python tools/train.py configs/0_ZUT_baseline/van_b2.py
python tools/train.py configs/0_ZUT_baseline/van_b3.py
python tools/train.py configs/0_ZUT_baseline/r50.py
python tools/train.py configs/0_ZUT_baseline/poolformer_s12.py
python tools/train.py configs/0_ZUT_baseline/convnext_tiny.py
```

## Citation

```bibtex
@inproceedings{Fairness_ZUT_WJPan,
   author = {Pan, Wenjie and Zhu, Jianqing and Zeng, Huanqiang},
   title = {Fair Training with Zero Inputs},
   booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
   volume = {39},
   pages = {6317-6325},
   address= {Pennsylvania, USA},
   year = {2025},
   type = {Conference Proceedings}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
