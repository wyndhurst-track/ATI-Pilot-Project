# Identification via Metric Learning

This repository contains the source code that accompanies our paper "Visual Identification of Individual Holstein Friesian Cattle via Deep Metric Learning" - available at [https://arxiv.org/abs/2006.09205](https://arxiv.org/abs/2006.09205) and [compag link](tbc).
At its core, the code in this repository is adapted and extended (with permission) from Lagunes-Fortiz, M. et al's work on "Learning Discriminative Embeddings for Object Recognition on-the-fly" published in ICRA 2019 - [paper](https://ieeexplore.ieee.org/document/8793715), [source code](https://github.com/MikeLagunes/Supervised-Triplet-Network).

Within our paper, this code relates to section 5 on the "Open-Set Individual Identification via Metric Learning" and the experiments conducted in section 6.

### Installation

Simply clone this repository to your desired local directory: `git clone tbc.git`
Install any missing requirements via `pip` or `conda`:

[numpy](https://pypi.org/project/numpy/)
[PyTorch](https://pytorch.org/)
[tqdm](https://pypi.org/project/tqdm/)
[sklearn](https://pypi.org/project/scikit-learn/)

### Usage

To replicate the results obtained in our paper, please download the OpenCows2020 dataset at: [https://www.data.bris.ac.uk/tbc](https://www.data.bris.ac.uk/tbc).
Trained network weights used in the paper are also provided in the dataset.

To train the model, use `python train.py -h` to get help with setting command line arguments

To test a trained model, use `python test.py -h` to get help with setting command line arguments

To visualise any embeddings, use ...

### Citation

Please consider citing ours and Miguel's works in your own research if this has been useful:
```
@article{andrew2020visual,
  title={Visual Identification of Individual Holstein Friesian Cattle via Deep Metric Learning},
  author={Andrew, William and Gao, Jing and Campbell, Neill and Dowsey, Andrew W and Burghardt, Tilo},
  journal={arXiv preprint arXiv:2006.09205},
  year={2020}
}

@inproceedings{lagunes2019learning,
  title={Learning discriminative embeddings for object recognition on-the-fly},
  author={Lagunes-Fortiz, Miguel and Damen, Dima and Mayol-Cuevas, Walterio},
  booktitle={2019 International Conference on Robotics and Automation (ICRA)},
  pages={2932--2938},
  year={2019},
  organization={IEEE}
}
```
