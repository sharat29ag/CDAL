# CDAL in PyTorch
PyTorch implementation of CDAL "Contextual Diversity for Active Learning" ECCV20.

Sharat Agarwal*, Himanshu Arora*, Saket Anand, Chetan Arora. 

*First two authors contributed equally*

Link to the paper: 

### Prerequisites:
* Python 3.6
* Pytorch >= 0.4.1
* CUDA 9.0 or higher
* CPU compatible but NVIDIA GPU + CUDA CuDNN is recommended.

### Installation
Clone the repo:
```bash
$ git clone https://github.com/sharat29ag/CDAL
$ cd CDAL
```
### Training
By default, logs are stored in ```<root_dir>/log``` with this structure:
```bash
<root_dir>/experiments/logs
```
Fo CDAL selection:
```bash
python main.py --number_of_picks <number of frames to select> --path_to_features <path to preprocessed features> --classes <number of classes in dataset> --gpu 1 --save-dir log/summe-split0 --start_idx 0
```
## Baseline Networks
- [Semantic Segmentation](https://github.com/fyu/drn)
- [Object Detection](https://github.com/amdegroot/ssd.pytorch)
- [Image Classification](https://github.com/kuangliu/pytorch-cifar)

## Acknowledgements
This codebase is borrwed from [VSUMM](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce)

## Contact
If there are any questions or concerns feel free to send a message at sharata@iiitd.ac.in

