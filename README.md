
## 1 Dependency
```bash
CUDA 12.1
Pytorch  2.1.0+cu121
PyYAML@5.4.1
scipy@1.13.1
nuscenes
open3d numpy
```

## 2 Training Process
### 2.1 Installation

调整gcc版本

`	sudo apt update`

`	sudo apt install gcc-12 g++-12`

`export CC=/usr/bin/gcc-12` 

`export CXX=/usr/bin/g++-12`

```
/home/tanzh/anaconda3/envs/ld/lib/python3.9/site-packages/torch/include/pybind11/cast.h
return caster;
```

```bash
python3 setup.py install
```

### 2.2 Data

SemanticSTF数据集，5个维度 （x,y,z,r,?）

### 2.3 Training Script

```bash
torchrun --master_port=12098 --nproc_per_node=4 train.py --config config/config_cpgnet_sgd_bili_sample_ohem_fp16.py
```

## 3 Evaluate Process
```bash
torchrun --master_port=12097 --nproc_per_node=4 evaluate.py --config config/config_cpgnet_sgd_bili_sample_ohem_fp16.py --start_epoch 0 --end_epoch 47
```

## 