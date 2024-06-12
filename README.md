# Open-Event Procedure Planning in Instructional Videos

*[Yilu WU]*<sup>1</sup>, 
*[Hanlin Wang]*<sup>1</sup>, 
*[Jing Wang]*<sup>1</sup>, 
*[Limin Wang]*<sup>1</sup>

<sup>1</sup>State Key Laboratory for Novel Software Technology, Nanjing University

<p align = "center"> 
<img src="img\\intro-sample.png"  width="600" />
</p>

**Abstract** : Given the current visual observations, the traditional procedure planning task in instructional videos requires a model to generate goal-directed plans within a given action space. 
All previous methods for this task conduct training and inference under the same action space, and they can only plan for pre-defined events in the training set. We argue this setting is not applicable for human assistance in real lives and aim to propose a more general and practical planning paradigm. 
Specifically, in this paper, we introduce a new task named Open-event Procedure Planning (OEPP), which extends the traditional procedure planning to the open-event setting. OEPP aims to verify whether a planner can transfer the learned knowledge to similar events that have not been seen during training. 
We rebuild a new benchmark of OpenEvent for this task based on existing datasets and divide the events involved into base and novel parts. During the data collection process, we carefully ensure the transfer ability of procedural knowledge for base and novel events by evaluating the similarity between the descriptions of different event steps with multiple stages. 
Based on the collected data, we further propose a simple and general framework specifically designed for OEPP, and conduct extensive study with various baseline methods, providing a detailed and insightful analysis on the results for this task.

## Dataset
Our data splits and annotations are under `data`.

| **File**                             | **Description**    |
|--------------------------------------|--------------------|
| data/train_train_base_dataset_1.json | train dataset      |
| data/train_train_val_dataset_1.json  | val dataset        |
| data/novel_dataset_1.json            | test novel dataset |
| data/test_base_dataset_1.json        | test base dataset  |
| data/base_action_pool_1.json         | base action pool   |
| data/novel_action_pool_1.json        | novel action pool  |
| data/total_action_pool_1.json        | total action pool  |
| data/task_info.json                  | event info         |

## Install Dependecny

`conda create --name oepp python=3.9`

`conda activate oepp`

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

`pip install -r requirements.txt`

## Download features

Download the features [link](https://drive.google.com/drive/folders/1IKrEnPhIvQhBN-tiIvtn_bG6EtDE8bNs?usp=drive_link)
and modify the path to the features in the `dataset/dataset.py Line51-62`.
Note that our videoclip features are currently only available for OEPP. We will release the complete features as soon as possible.

## Training

### MLP-based 
You can run the code using the following command, and the results will be in `results/MLP`.

`python train.py --config=MLP_config.yaml`

### Transformer-based 

You can run the code using the following command, and the results will be in `results/attention`.

`python train.py --config=attention_config.yaml`

### PDPP*

You can run the code using the following command, and the results will be in `log`. You can use TensorBoard to view the results.

`CUDA_VISIBLE_DEVICES=0 python pdpp_train.py --multiprocessing-distributed --num_thread_reader=1 --cudnn_benchmark=1 --pin_memory --checkpoint_dir=whl --resume --dist-url='tcp://localhost:21723' --horizon=3 --feat='videoclip' --split=1 --para_mse=0.2 --para_ce=1.0 --lr=0.0005 --batch_size=32 --batch_size_val=32 --evaluate`