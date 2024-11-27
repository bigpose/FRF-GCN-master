# FRF-GCN-master
_Behavioral Recognition of Skeletal Data Based on Targeted  Dual Fusion Strategy in AAAI2024._
The overall structure of FRF-GCN is shown below.

![image](https://github.com/sunbeam-kkt/FRF-GCN-master/blob/main/The%20main%20architecture.png)
# Abstract
The deployment of multi-stream fusion strategy on behavioral recognition from skeletal data can extract complementary features from different information streams and improve the recognition accuracy, but suffers from high model complexity and a large number  of parameters. Besides, existing multi-stream methods using a fixed adjacency matrix homogenizes the model’s discrimination process across diverse actions, causing reduction of the actual lift for the multi-stream model. Finally, attention mechanisms are commonly applied to the multi-dimensional features, including spatial, temporal and channel dimensions. But their attention scores are typically fused in a concatenated manner, leading to the ignorance of the interrelation between joints in complex actions. To alleviate these issues, the Front-Rear dual Fusion Graph Convolutional Network (FRF-GCN) is proposed to provide a lightweight model based on skeletal data. Targeted adjacency matrices are also designed for different front fusion streams, allowing the model to focus on actions of varying magnitudes. Simultaneously, the mechanism of Spatial-Temporal-Channel Parallel Attention (STC-P), which processes attention in parallel and places greater emphasis on useful information, is proposed to further improve model’s performance. FRF-GCN demonstrates significant competitiveness compared to the current state-of-the-art methods on the NTU RGB+D, NTU RGB+D 120 and Kinetics-Skeleton 400 datasets. 
# Prerequisites
- Python3(≥3.8)
- PyTorch
- Other Python libraries can be installed by `pip install -r requirements.txt`
# Data Preparation
Download the raw data from [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D) and [Skeleton-Kinetics](https://github.com/yysijie/st-gcn). Then put them under the data directory:
- data/
  - nturgbd_raw\
    - nturgb+d_skeletons\
    - samples_with_missing_skeletons.txt
  - kinetics_raw\
    - kinetics_train\
    - kinetics_val\
    - kinetics_train_label.json
    - keintics_val_label.json

Processes data and generates skeletal joint data.

```
python data_gen/ntu_gendata.py
```

```
python data_gen/kinetics-gendata.py
```

Generate the bone data with:

```
python data_gen/gen_bone_data.py
```

```
python data_gen/kinetics_gen_bone_data.py
```

Generate the motion data with:

```
python data_gen/gen_motion_data.py
```

```
python data_gen/kinetics_gen_motion_data.py
```

Forward fusion of data is performed by the following command：

```
python data_gen/merge_joint_bone_data.py
```

```
python data_gen/merge_joint_bone_motion_data.py
```

```
python data_gen/kinetics_merge_joint_bone.py
```

```
python data_gen/kinetics_merge_joint_bone_motion.py
```

# Training & Testing

Train the model according to your needs by modifying the configuration file.When training different forward fusion flow information, the selection of the targeted adjacency matrix only requires modifying lines **319-322** and the corresponding lines **356-358** in the `model/agcn_stc_sl.py` file.

Train and test the model with the following commands：

```
python main.py --config ./config/nturgbd-cross-subject/train_joint_bone.yaml
```

```
python main.py --config ./config/nturgbd-cross-subject/train_joint_bone_motion.yaml
```

```
python main.py --config ./config/nturgbd-cross-subject/test_joint_bone.yaml
```

```
python main.py --config ./config/nturgbd-cross-subject/test_joint_bone_motion.yaml
```

```
python main.py --config ./config/kinetics-skeleton/train_joint_bone.yaml
```

```
python main.py --config ./config/kinetics-skeleton/train_joint_bone_motion.yaml
```

```
python main.py --config ./config/kinetics-skeleton/test_joint_bone.yaml
```

```
python main.py --config ./config/kinetics-skeleton/test_joint_bone_motion.yaml
```

# Ensemble

The resulting test scores are fused by post-fusion, which is performed by the following command:

```
python ensemble.py
```
# Acknowledgements

Our FRF-GCN has been inspired by previous research, and we thank the authors of the following studies for their significant contributions!
- [ST-GCN](https://github.com/yysijie/st-gcn)
- [2s-AGCN](https://github.com/lshiwjx/2s-AGCN)
- [EfficientGCN](https://github.com/yfsong0709/EfficientGCNv1)

# Citation

If you find this work helpful, please cite our work:
```
Yun, X., Xu, C., Riou, K., Dong, K., Sun, Y., Li, S., Subrin, K., & Le Callet, P. (2024).
Behavioral Recognition of Skeletal Data Based on Targeted Dual Fusion Strategy.
Proceedings of the AAAI Conference on Artificial Intelligence,
38(7), 6917-6925. https://doi.org/10.1609/aaai.v38i7.28517
```
# Contact

For any question, feel free to contact Chenglong Xu: ```clongxu@cumt.edu.cn```
