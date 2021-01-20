# CDAL in PyTorch
PyTorch implementation of CDAL "Contextual Diversity for Active Learning" accepted in ECCV20.

Sharat Agarwal, Himanshu Arora, Saket Anand, Chetan Arora 

*First two authors contributed equally*
[Link to the paper](https://arxiv.org/pdf/2008.05723.pdf) 


## Citation
If using this code, parts of it, or developments from it, please cite our paper:
```
@inproceedings{agarwal2020contextual,
  title={Contextual Diversity for Active Learning},
  author={Agarwal, Sharat and Arora, Himanshu and Anand, Saket and Arora, Chetan},
  booktitle={European Conference on Computer Vision},
  pages={137--153},
  year={2020},
  organization={Springer}
}
```
## Proposed Architecture
![](./images/work_flow.jpg)

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
### Frame Selection
By default, logs are stored in ```<root_dir>/log``` with this structure:
```bash
<root_dir>/experiments/logs
```
Sample features in features folder for PASCAL-VOC.

For weighted features:
```bash
python preprocess.py
```
Change the path to raw features in the preprocess.py

Creates a folder <root_dir>/features2 with weighted features.

For CDAL-RL selection:
```bash
python main.py --number_of_picks <number of frames to select> --path_to_features <path to weighted features> --classes <number of classes in dataset> --gpu 1 --save-dir log/summe-split0 --start_idx 0
```
List of selected samples will be stored in <root_dir>/selection/

## Base Networks
- [Semantic Segmentation](https://github.com/fyu/drn)
- [Object Detection](https://github.com/amdegroot/ssd.pytorch)
- [Image Classification](https://github.com/kuangliu/pytorch-cifar)

## Acknowledgements
This codebase is borrwed from [VSUMM](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce)

## Contact
If there are any questions or concerns feel free to send a message at sharata@iiitd.ac.in

