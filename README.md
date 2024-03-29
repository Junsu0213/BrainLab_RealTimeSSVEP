# Integrated Wheelchair and Robotic Arm Control System for Enhancing Functional Independence of the Physically Disabled

<div align=center>

![flow_chart](https://github.com/Junsu0213/BrainLab_RealTimeSSVEP/assets/128777619/6666b5e9-7723-4c2b-a77d-e1edb0ce9172)

Flow chart of the proposed AR-based BCI-SSVEP integrated system for wheelchair and robotic arm

##### [DOI Link,](https://dcollection.korea.ac.kr/srch/srchDetail/000000278021) [Youtube Link](https://youtu.be/hDDy_OVMIyI)

</div>

## 1. Abstract
> Recent advances in biosignal processing and robotics have enabled the operation of wheelchairs and robotic arms using electroencephalography (EEG) through brain-computer interfaces (BCI). However, most BCI studies rely on monitor screens, potentially limiting usability. To address these issues, this study integrates augmented reality (AR), eye tracking, spatial mapping, and object detection technologies to develop and implement a wheelchair and robotic arm integrated operation system that utilizes steady-state visual evoked potential (SSVEP), a major area of research in BCI. The system uses HoloLens 2 to induce SSVEP in an AR environment, which has the advantage of allowing users to operate the device more intuitively without the need to switch their gaze.
> 
> To evaluate the effectiveness of the system, a single case study with healthy participants was used to assess performance and usability. The participant selected a control device using AR-based eye-tracking technology. A SSVEP with four commands (‘Forward’, ‘Backward’, ‘Turn Left’, and ‘Turn Right’) was used for wheelchair operation. For robotic arm control, an SSVEP with three commands (‘Bring’, ‘Move’, and ‘Cancel’) was presented post cup-grasping via object detection and spatial mapping; if 'Move' was selected, a four-command SSVEP (‘Pull’, ‘Push’, ‘Move Left’, and ‘Move Right’) enabled cup movement.
> 
> Online experiment results demonstrated the successful operation of wheelchairs and robotic arms using the system, achieving classification accuracies of 100% for the three-class model and 90% for the four-class model.
> 
> These results indicate that the AR-based BCI has high potential in the control of wheelchairs and robotic arms, thus contributing significantly to improving the functional independence of patients with physical disabilities in the future.

## 2. Installation

#### Environment
* Python == 2.8.2
* PyTorch == 1.9.1+cu111
* MNE == 1.3.1
* brainflow == 5.6.1

## 3. Directory structure
```bash
├── Config
│   ├── data_config.py
│   ├── model_config.py
│   └── train_config.py
├── Evaluation
│   └── model_evaluation.py
├── Figure
│   ├── confusion_matrix.py
│   └── ssvep_spectrum_tf.py
├── Loader
│   └── data_epoching.py
├── Model
│   ├── Base
│   │   └── Layer.py
│   ├── DeepConvNet
│   │   └── DeepConvNet_model.py
│   ├── EEGNet
│   │   └── EEGNet_model.py
│   ├── FBCCA
│   │   └── FBCCA_model.py
│   ├── FBCSP
│   │   └── FBCSP_model.py
│   ├── Riemann
│   │   └── riemann_model.py
│   ├── ShallowConvNet
│   │   └── ShallowConvNet_model.py
│   └── Trainer
│       ├── k_fold_brainlab_main.py
│       ├── k_fold_openbmi_main.py
│       ├── loso_brainlab_main.py
│       ├── loso_openbmi_main.py
│       └── model_trainer.py
├── Real_time
│   ├── SSVEP_data_aqusition.py
│   ├── classification.py
│   ├── open_bci.py
│   └── todo.py
└── requirements.txt
```

## 4. Dataset

#### Brain Lab. SSVEP dataset
* 8 subjects
* Classes: 5.45 Hz, 6.67 Hz, 8.57 Hz, 12 Hz (4 classes)
* Total trials: 320 trials (80 trials per target)

#### Preprocessing
* Sampling rate: 125 Hz
* Time segment: [-0.5, 5] sec
* Band-pass filtering: 0.5~50 Hz
* Normalization: RobustScaler
