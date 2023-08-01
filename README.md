# FRF-GCN-master
The overall structure of FRF-GCN is shown below.
![image](https://github.com/sunbeam-kkt/FRF-GCN-master/assets/117554619/57ceaabf-7cbe-45be-936b-e0e9f72b92c2)
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

`python data_gen/ntu_gendata.py`

`python data_gen/kinetics-gendata.py`

Generate the bone data with:

`python data_gen/gen_bone_data.py`

`python data_gen/kinetics_gen_bone_data.py`

Generate the motion data with:

`python data_gen/gen_motion_data.py`

`python data_gen/kinetics_gen_motion_data.py`

Forward fusion of data is performed by the following command：

·python data_gen/merge_joint_bone_data.py`

`python data_gen/merge_joint_bone_motion_data.py`

`python data_gen/kinetics_merge_joint_bone.py`

`python data_gen/kinetics_merge_joint_bone_motion.py`

# Training & Testing

Train the model according to your needs by modifying the configuration file.Train and test the model with the following commands：

`python main.py --config ./config/nturgbd-cross-subject/train_joint_bone.yaml`

`python main.py --config ./config/nturgbd-cross-subject/train_joint_bone_motion.yaml`

`python main.py --config ./config/nturgbd-cross-subject/test_joint_bone.yaml`

`python main.py --config ./config/nturgbd-cross-subject/test_joint_bone_motion.yaml`

`python main.py --config ./config/kinetics-skeleton/train_joint_bone.yaml`

`python main.py --config ./config/kinetics-skeleton/train_joint_bone_motion.yaml`

`python main.py --config ./config/kinetics-skeleton/test_joint_bone.yaml`

`python main.py --config ./config/kinetics-skeleton/test_joint_bone_motion.yaml`

# Ensemble

The resulting test scores are fused by post-fusion, which is performed by the following command:

`python ensemble.py`
