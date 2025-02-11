# A 3D Multimodal Feature for Infrastructure Anomaly Detection

This algorithm is designed to detect structural defects by leveraing anomaly detection from infrastructure point clouds. For simplicity, we integrated all functions from our previous paper: [**Anomaly detection of cracks in synthetic masonry arch bridge point clouds using fast point feature histograms and PatchCore**](https://www.sciencedirect.com/science/article/pii/S0926580524005028) into this repository.

## Authors
- [Yixiong Jing](https://www.researchgate.net/profile/Yixiong_Jing2), [Wei Lin](https://www.researchgate.net/profile/Wei-Lin-126), [Brian Sheil](https://www.construction.cam.ac.uk/staff/dr-brian-sheil), [Sinan Acikgoz](https://eng.ox.ac.uk/people/sinan-acikgoz/)

## Setup
This code has been tested with Python 3.8, CUDA 11.8, and Pytorch 2.0.1 on Ubuntu 18.04. FPFH is comptued with 128GB of memory. [CPMF](https://github.com/caoyunkang/CPMF) is tested on RTX3080.

```bash
  conda create -n infra_inspect python=3.8
  conda activate infra_inspect
  pip install -r requirements.txt
```

## Dataset 
All datasets are available in [here](https://huggingface.co/datasets/jing222/infra_3DAL/tree/main). Please download the data.zip file and unzip all datasets in the `\infra_3DALv2` path for replicating our results.

## Usage
The algorithm can be used for computing anomalies on large-scale infrastructure point clouds.

#### 1. Generate downsampled point clouds(voxelization) and images(projected from 3D to 2D) 
- Run:
```python
  python multi_view_main.py 
```

#### 2. Evaluate on synthetic masonry arch point clouds

- (1) All synthetic masonry arch point clouds
```python
  python main.py '++general.inspect_target="syn_arch"'
```
- (2) Only on different support movement cases
```python
  python main.py 
  '++general.inspect_target="syn_arch"' 
  '++general.synarch_names=["disp_x_40cm", "disp_z", "disp_xz", "rot_x"]'
```
- (3) Only on varying support movement magnitude(It is not included in this paper for the length limit. Though it is a good demonstration for comparing the difference on whether or not adding new surfaces on the synthetic dataset.)
```python
  python main.py 
  '++general.inspect_target="syn_arch"' 
  '++general.synarch_names=["disp_x_8cm", "disp_x_12cm", "disp_x_8cm_noinnerc", "disp_x_12cm_noinnerc"]'
```

#### 3. Evaluate on real masonry arch point clouds
- Run:
```python
  python main.py '++general.inspect_target="real_arch"'
```
#### 4. Evaluate on real tunnel point clouds 
- Run:
```python
  python main.py 
  '++general.inspect_target="tunnel"'
  '++al_detector.feature_types=["FPFH", "FPFH_naiveRGB", "FPFH_relaRGB"]'
  '++al_detector.radius_fs_ratios=[30]'
```

## Results

Comparison of different feature types in anomaly detection:

#### 1. Synthetic masonry arch
![Results](/img/syn_result.jpg)

#### 2. Real masonry arch
![Results](/img/realarch_result.jpg)

#### 3. Real tunnel
![Results](/img/realtunnel_result.jpg)

## Citations

If you find the code is beneficial to your research, please consider citing:

## Acknowledge
We used some code from [CPMF](https://github.com/caoyunkang/CPMF) to make comparison in our work. We would like to appreciate for their sharing.

## License
Our work is subjected to MIT License.
