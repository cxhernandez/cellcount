# cellcount

This is a Python 3.4+ project that uses PyTorch v0.4.1.

## Usage

After installing via `python setup.py install`, you can use the command-line commands to download an example BBBC dataset, train the feature pyramid network (FPN), and finally train the fulll end-to-end cell counting network:

```
$ cell_count download --dataset bbbc005
$ cell_count train_fpn --dataset path/to/bbbc005
$ cell_count train --dataset path/to/bbbc005
```

If you find this software useful for your work, please cite:

```bibtex
@article{cellcount,
Author = {Carlos X. Hernández and Mohammad M. Sultan and Vijay S. Pande},
Title = {Using Deep Learning for Segmentation and Counting within Microscopy Data},
Year = {2018},
Eprint = {arXiv:1802.10548},
}
```
