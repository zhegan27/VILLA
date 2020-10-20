# VILLA: Vision-and-Language Adversarial Training
This is the official repository of [VILLA](https://arxiv.org/pdf/2006.06195.pdf) (NeurIPS 2020 Spotlight).
This repository currently supports adversarial finetuning of UNITER on [VQA](https://visualqa.org/), [VCR](https://visualcommonsense.com/), [NLVR2](http://lil.nlp.cornell.edu/nlvr/), and 
[SNLI-VE](https://github.com/necla-ml/SNLI-VE).
Adversarial pre-training with in-domain data will be available soon.
Both VILLA-base and VILLA-large pre-trained checkpoints are released. 

![Overview of VILLA](villa_framework.png)

Most of the code in this repo are copied/modified from [UNITER](https://github.com/ChenRocks/UNITER).


## Requirements
We provide Docker image for easier reproduction. Please install the following:
  - [nvidia driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) (418+), 
  - [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) (19.03+), 
  - [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart).

Our scripts require the user to have the [docker group membership](https://docs.docker.com/install/linux/linux-postinstall/)
so that docker commands can be run without sudo.
We only support Linux with NVIDIA GPUs. We test on Ubuntu 18.04 and V100 cards.
We use mixed-precision training hence GPUs with Tensor Cores are recommended.

## Quick Start
*NOTE*: Please run `bash scripts/download_pretrained.sh $PATH_TO_STORAGE` to get our latest pretrained VILLA
checkpoints. This will download both the base and large models.

We use VQA as an end-to-end example for using this code base.

1. Download processed data and pretrained models with the following command.
    ```bash
    bash scripts/download_vqa.sh $PATH_TO_STORAGE
    ```
    After downloading you should see the following folder structure:
    ```
    ├── finetune 
    ├── img_db
    │   ├── coco_test2015
    │   ├── coco_test2015.tar
    │   ├── coco_train2014
    │   ├── coco_train2014.tar
    │   ├── coco_val2014
    │   ├── coco_val2014.tar
    │   ├── vg
    │   └── vg.tar
    ├── pretrained
        ├── uniter-base.pt
    │   └── villa-base.pt
    └── txt_db
        ├── vqa_devval.db
        ├── vqa_devval.db.tar
        ├── vqa_test.db
        ├── vqa_test.db.tar
        ├── vqa_train.db
        ├── vqa_train.db.tar
        ├── vqa_trainval.db
        ├── vqa_trainval.db.tar
        ├── vqa_vg.db
        └── vqa_vg.db.tar

    ```
    You can put different pre-trained checkpoints inside the /pretrained folder based on your need. 

2. Launch the Docker container for running the experiments.
    ```bash
    # docker image should be automatically pulled
    source launch_container.sh $PATH_TO_STORAGE/txt_db $PATH_TO_STORAGE/img_db \
        $PATH_TO_STORAGE/finetune $PATH_TO_STORAGE/pretrained
    ```
    The launch script respects $CUDA_VISIBLE_DEVICES environment variable.
    Note that the source code is mounted into the container under `/src` instead 
    of built into the image so that user modification will be reflected without
    re-building the image. (Data folders are mounted into the container separately
    for flexibility on folder structures.)


3. Run finetuning for the VQA task.
    ```bash
    # inside the container
    horovodrun -np $N_GPU python train_vqa_adv.py --config $YOUR_CONFIG_JSON

    # specific example
    horovodrun -np 4 python train_vqa_adv.py --config config/train-vqa-base-4gpu-adv.json
    ```

4. Run inference for the VQA task and then evaluate.
    ```bash
    # inference
    python inf_vqa.py --txt_db /txt/vqa_test.db --img_db /img/coco_test2015 \
    --output_dir $VQA_EXP --checkpoint 6000 --pin_mem --fp16
    ```
    The result file will be written at `$VQA_EXP/results_test/results_6000_all.json`, which can be
    submitted to the evaluation server


5. Customization
    ```bash
    # training options
    python train_vqa_adv.py --help
    ```
    - command-line argument overwrites JSON config files
    - JSON config overwrites `argparse` default value.
    - use horovodrun to run multi-GPU training
    - `--gradient_accumulation_steps` emulates multi-gpu training
    - `--checkpoint` selects UNITER or VILLA pre-trained checkpoints
    - `--adv_training` decides using adv. training or not
    - `--adv_modality` takes values from ['text'], ['image'], ['text','image'], and ['text','image','alter'], the last two correspond to adding perturbations on two modalities simultaneously or alternatively

## Downstream Tasks Finetuning

### VCR
NOTE: train and inference should be ran inside the docker container
1. download data
    ```
    bash scripts/download_vcr.sh $PATH_TO_STORAGE
    ```
2. train
    ```
    horovodrun -np 4 python train_vcr_adv.py --config config/train-vcr-base-4gpu-adv.json \
        --output_dir $VCR_EXP
    ```
3. inference
    ```
    horovodrun -np 4 python inf_vcr.py --txt_db /txt/vcr_test.db \
        --img_db "/img/vcr_gt_test/;/img/vcr_test/" \
        --split test --output_dir $VCR_EXP --checkpoint 8000 \
        --pin_mem --fp16
    ```
    The result file will be written at `$VCR_EXP/results_test/results_8000_all.csv`, which can be
    submitted to VCR leaderboard for evaluation.

### NLVR2
NOTE: train and inference should be ran inside the docker container
1. download data
    ```
    bash scripts/download_nlvr2.sh $PATH_TO_STORAGE
    ```
2. train
    ```
    horovodrun -np 4 python train_nlvr2_adv.py --config config/train-nlvr2-base-1gpu-adv.json \
        --output_dir $NLVR2_EXP
    ```
3. inference
    ```
    python inf_nlvr2.py --txt_db /txt/nlvr2_test1.db/ --img_db /img/nlvr2_test/ \
    --train_dir /storage/nlvr-base/ --ckpt 6500 --output_dir . --fp16
    ```

### Visual Entailment (SNLI-VE)
NOTE: train should be ran inside the docker container
1. download data
    ```
    bash scripts/download_ve.sh $PATH_TO_STORAGE
    ```
2. train
    ```
    horovodrun -np 2 python train_ve_adv.py --config config/train-ve-base-2gpu-adv.json \
        --output_dir $VE_EXP
    ```

## Adversarial Training of LXMERT

To keep things simple, we provide [another separate repo](https://github.com/zhegan27/LXMERT-AdvTrain) that can be used to reproduce our results on adversarial finetuning of [LXMERT](https://arxiv.org/pdf/1908.07490.pdf) on [VQA](https://visualqa.org/), [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html), and [NLVR2](http://lil.nlp.cornell.edu/nlvr/).

## Citation

If you find this code useful for your research, please consider citing:
```
@inproceedings{gan2020large,
  title={Large-Scale Adversarial Training for Vision-and-Language Representation Learning},
  author={Gan, Zhe and Chen, Yen-Chun and Li, Linjie and Zhu, Chen and Cheng, Yu and Liu, Jingjing},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{chen2020uniter,
  title={Uniter: Universal image-text representation learning},
  author={Chen, Yen-Chun and Li, Linjie and Yu, Licheng and Kholy, Ahmed El and Ahmed, Faisal and Gan, Zhe and Cheng, Yu and Liu, Jingjing},
  booktitle={ECCV},
  year={2020}
}
```

## License

MIT
