# Group-Aware-Hierarchical-Transformer

This repository is the official implementation for our IEEE TGRS 2022 paper:

[Hyperspectral image classification using group-aware hierarchical transformer](https://www.doi.org/10.1109/TGRS.2022.3207933)

Last update: September 20, 2022

## Requirements

python == 3.7.9, cuda == 11.1, and packages in `requirements.txt`

## Datasets

Download following datasets:

- [Salinas (SA)](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)

- [Pavia University (PU)](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)

- [WHU-LongKou (WHU-LK)](http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm)

- [HyRANK-Loukia (HR-L)](https://zenodo.org/record/1222202#.Y4HMrX1Bxdi)

Then organize these datasets like:

```
datasets/
  hrl/
    Loukia_GT.tif
    Loukia.tif
  pu/
    PaviaU_gt.mat
    PaviaU.mat
  sa/
    Salinas_corrected.mat
    Salinas_gt.mat
  whulk/
    WHU_Hi_LongKou_gt.mat
    WHU_Hi_LongKou.mat
```

## Codes for Training and Validation

Train our proposed GAHT using train-val-test split ratios in the paper: 

### For the SA/PU/WHU-LK Dataset:

```bash
python main.py --model proposed --dataset_name sa --epoch 300 --bs 64 --device 0 --ratio 0.02
python main.py --model proposed --dataset_name pu --epoch 300 --bs 64 --device 0 --ratio 0.02
python main.py --model proposed --dataset_name whulk --epoch 300 --bs 64 --device 0 --ratio 0.01
```

### For the HRL Dataset:

Transform the format of HRL dataset first:

```bash
python utils/tif2mat.py
```

Then train the model like other datasets: 

```bash
python main.py --model proposed --dataset_name hrl --epoch 300 --bs 64 --device 0 --ratio 0.06
```

## Evaluate the Model

```bash
python eval.py --model proposed --dataset_name sa --device 0 --weights ./checkpoints/proposed/sa/0
python eval.py --model proposed --dataset_name pu --device 0 --weights ./checkpoints/proposed/pu/0
python eval.py --model proposed --dataset_name whulk --device 0 --weights ./checkpoints/proposed/whulk/0
python eval.py --model proposed --dataset_name hrl --device 0 --weights ./checkpoints/proposed/hrl/0
```

## Other Supported SOTA Methods:

| Method                                                       | Abbr.    | Parameter         | Paper                                                 |
| ------------------------------------------------------------ | -------- | ----------------- | ----------------------------------------------------- |
| multi-scale3D deep convolutional neural network              | M3D-DCNN | --model m3ddcnn   | [here](https://ieeexplore.ieee.org/document/8297014)  |
| CNN-based 3D deep learning approach                          | 3D-CNN   | --model cnn3d     | [here](https://ieeexplore.ieee.org/document/8344565/) |
| deep feature fusion network                                  | DFFN     | --model dffn      | [here](https://ieeexplore.ieee.org/document/8283837)  |
| residual spectral-spatial attention network                  | RSSAN    | --model rssan     | [here](https://ieeexplore.ieee.org/document/9103247)  |
| attention-based bidirectional long short-term memory network | AB-LSTM  | --model ablstm    | [here](https://ieeexplore.ieee.org/document/9511338)  |
| transformer-based backbone network                           | SF       | --model speformer | [here](https://ieeexplore.ieee.org/document/9627165)  |
| spectral–spatial feature tokenization transformer            | SSFTT    | --model ssftt     | [here](https://ieeexplore.ieee.org/document/9684381)  |

## Citation

Please cite our paper if our work is helpful for your research.

```
@article{gaht,
  title={Hyperspectral image classification using group-aware hierarchical transformer},
  author={Mei, Shaohui and Song, Chao and Ma, Mingyang and Xu, Fulin},
  journal={IEEE Trans. Geosci. Remote Sens.},
  year={2022},
  volume={60},
  pages={1-14},
  doi={10.1109/TGRS.2022.3207933}}
```

## Acknowledgement

Some of our codes references to the following projects, and we are thankful for their great work:

- [HPDM-SPRN](https://github.com/shangsw/HPDM-SPRN)

- [DeepHyperX](https://github.com/nshaud/DeepHyperX)
