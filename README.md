# Backdoor Attacks Against Dataset Distillation

[![arXiv](https://img.shields.io/badge/arxiv-2301.01197-b31b1b)](https://arxiv.org/abs/2301.01197)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

This is official code of our NDSS 23 paper Backdoor Attacks Against Dataset Distillation.
Currently, we apply two distillation techniques, namely [Dataset Distillation (DD)](https://arxiv.org/pdf/1811.10959.pdf) and [Dataset Condensation with Gradient Matching (DC)](https://arxiv.org/pdf/2006.05929.pdf).
In the project, we propose three different backdoor attacks, *NAIVEATTACK*, *DOORPING*, and *INVISIBLE*.
NAIVEATTACK inserts a pre-defined trigger into the original training dataset before the distillation.
DOORPING is an advanced method, which optimizes the trigger during the distillation process.
Required by the reviewers, we need to add another backdoor method.
So, we choose [Invisible Backdoor Attacks on Deep Neural Networks via Steganography and Regularization](https://arxiv.org/pdf/1909.02742.pdf).

Limited by the [DD code](https://github.com/SsnL/dataset-distillation), [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/) are not supported.

## Requirments
A suitable [conda](https://conda.io/) environment named `baadd` can be created and activated with:

```
conda env create -f environment.yaml
conda activate baadd
```

## Run Backdoor Attacks against DD
We support five different dataset: Fashion-MNIST (FMNIST), CIFAR10, CIFAR100, STL10, and SVHN.
And two attack architectures: AlexNet and ConvNet

Due to the different arguments between DD and DC code, we list the arguments in the following:

<table><tbody>
<!-- TABLE BODY -->
<tr>
<td align="center">Dataset Name</td>
<td align="center">Fashion-MNIST</td>
<td align="center">CIFAR10</td>
<td align="center">CIFAR100</td>
<td align="center">STL10</td>
<td align="center">SVHN</td>
</tr>
<tr>
<td align="center">Arguments</td>
<td align="center">FashionMNIST</td>
<td align="center">Cifar10</td>
<td align="center">Cifar100</td>
<td align="center">STL10</td>
<td align="center">SVHN</td>
</tr>
</tbody></table>

<table><tbody>
<!-- TABLE BODY -->
<tr>
<td align="center">Model Architecture</td>
<td align="center">AlexNet</td>
<td align="center">ConvNet</td>
</tr>
<tr>
<td align="center">Arguments</td>
<td align="center">AlexCifarNet</td>
<td align="center">ConvNet</td>
</tr>
</tbody></table>

For NAIVEATTACK, run this mode via

```
python DD/main.py --mode distill_basic --dataset Cifar10 --arch AlexCifarNet --distill_lr 0.001 --naive --dataset_root /path/to/data --results_dir /path/to/results
```

For DOORPING, run this mode via

```
python DD/main.py --mode distill_basic --dataset Cifar10 --arch AlexCifarNet --distill_lr 0.001 --doorping --dataset_root /path/to/data --results_dir /path/to/results
```

For INVISIBLE, run this mode via

```
python DD/main.py --mode distill_basic --dataset Cifar10 --arch AlexCifarNet --distill_lr 0.001 --invisible --dataset_root /path/to/data --results_dir /path/to/results
```

## Run Backdoor Attacks against DC

<table><tbody>
<!-- TABLE BODY -->
<tr>
<td align="center">Dataset Name</td>
<td align="center">Fashion-MNIST</td>
<td align="center">CIFAR10</td>
<td align="center">CIFAR100</td>
<td align="center">STL10</td>
<td align="center">SVHN</td>
</tr>
<tr>
<td align="center">Arguments</td>
<td align="center">FashionMNIST</td>
<td align="center">CIFAR10</td>
<td align="center">CIFAR100</td>
<td align="center">STL10</td>
<td align="center">SVHN</td>
</tr>
</tbody></table>

<table><tbody>
<!-- TABLE BODY -->
<tr>
<td align="center">Model Architecture</td>
<td align="center">AlexNet</td>
<td align="center">ConvNet</td>
</tr>
<tr>
<td align="center">Arguments</td>
<td align="center">AlexNet</td>
<td align="center">ConvNet</td>
</tr>
</tbody></table>

For NAIVEATTACK, run this mode via

```
python DC/main.py --dataset CIFAR10 --model AlexNet --naive --data_path /path/to/data --save_path /path/to/results
```

For DOORPING, run this mode via

```
python DC/main.py --dataset CIFAR10 --model AlexNet --doorping --data_path /path/to/data --save_path /path/to/results
```

For INVISIBLE, run this mode via

```
python DC/main.py --dataset CIFAR10 --model AlexNet --invisible --data_path /path/to/data --save_path /path/to/results
```

## Citation
Please cite this paper in your publications if it helps your research:

    @inproceedings{LLBSZ23,
    author = {Yugeng Liu and Zheng Li and Michael Backes and Yun Shen and Yang Zhang},
    title = {{Backdoor Attacks Against Dataset Distillation}},
    booktitle = {{NDSS}},
    year = {2023}
    }



## License

Baadd is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail at yugeng.liu@cispa.de. We will send the detail agreement to you.