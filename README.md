<div align="center">

## [ICML 2024] Parameter-Efficient Fine-Tuning with Controls

### [Paper](https://openreview.net/pdf?id=C4nalr0DoE) |  [Blog](https://medium.com/@czhang024/the-mechanism-behind-lora-a-control-view-f2079bba74b8)

<!-- ![teaser](figures/Arch.jpg) -->
<img src="figures/Arch.jpg" alt="teaser" style="width: 80%;">

</div>


This is a PyTorch implementation of the paper [Parameter-Efficient Fine-Tuning with Controls](https://openreview.net/pdf?id=C4nalr0DoE). The goal of our work is to provide a pure control view for the well-known LoRA algorithm.


### Usage

#### Install
* CUDA 11.2 + PyTorch 2.1.0 + torchvision 0.16.0
* timm 1.0.7
* easydict

####  Download Pretrained Model
The mae_pretrain_vit_b model is available [here](https://github.com/ShoufaChen/AdaptFormer/blob/main/PRETRAIN.md).

#### Training
Start
```bash
# video
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch \
    --nproc_per_node=8 --nnodes=8 \
    --node_rank=$1 --master_addr=$2 --master_port=22234 \
    --use_env main_video.py \
    --finetune /path/to/pre_trained/checkpoints \
    --output_dir /path/to/output \
    --batch_size 16 --epochs 90 --blr 0.1 --weight_decay 0.0 --dist_eval \
    --data_path /path/to/SSV2 --data_set SSV2 \
    --ffn_adapt
```
on each of 8 nodes. `--master_addr` is set as the ip of the node 0. and `--node_rank` is 0, 1, ..., 7 for each node.

```bash
# image
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_image.py \
    --batch_size 128 --cls_token \
    --finetune /path/to/pre_trained/mae_pretrain_vit_b.pth \
    --dist_eval --data_path /path/to/data \
    --output_dir /path/to/output  \
    --drop_path 0.0  --blr 0.1 \
    --dataset cifar100 --ffn_adapt
```

To obtain the pre-trained checkpoint, see [PRETRAIN.md](PRETRAIN.md).
### Acknowledgement

The project is based on [MAE](https://github.com/facebookresearch/mae) and [AdaptFormer](https://github.com/ShoufaChen/AdaptFormer).
Thank all the authors for their awesome works.

### Citation
```
@inproceedings{zhangparameter,
  title={Parameter-Efficient Fine-Tuning with Controls},
  author={Zhang, Chi and Jingpu, Cheng and Xu, Yanyu and Li, Qianxiao},
  booktitle={Forty-first International Conference on Machine Learning}
}
```

### License

This project is under the MIT license. See [LICENSE](LICENSE) for details.
