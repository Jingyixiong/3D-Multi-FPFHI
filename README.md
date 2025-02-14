# [A 3D Multimodal Feature for Infrastructure Anomaly Detection](https://arxiv.org/abs/2502.05779)

This paper proposed a method to detect structural defects by leveraging anomaly detection from infrastructure point clouds. For simplicity, we integrated all functions from our previous paper: [**Anomaly detection of cracks in synthetic masonry arch bridge point clouds using fast point feature histograms and PatchCore**](https://www.sciencedirect.com/science/article/pii/S0926580524005028) into this repository.

## Authors
- [Yixiong Jing](https://www.researchgate.net/profile/Yixiong_Jing2), [Wei Lin](https://www.researchgate.net/profile/Wei-Lin-126), [Brian Sheil](https://www.construction.cam.ac.uk/staff/dr-brian-sheil), [Sinan Acikgoz](https://eng.ox.ac.uk/people/sinan-acikgoz/)

## Setup
This code has been tested with Python 3.8, CUDA 11.8, and Pytorch 2.0.1 on Ubuntu 18.04. FPFH is computed with 128GB of memory. [CPMF](https://github.com/caoyunkang/CPMF) is tested on RTX3080.

```bash
  conda create -n infra_inspect python=3.8
  conda activate infra_inspect
  pip install -r requirements.txt
```

## Dataset 
All datasets are available [here](https://huggingface.co/datasets/jing222/infra_3DAL/tree/main). Please download the data.zip file and unzip all datasets in the `\infra_3DALv2` path for replicating our results.

## Usage
The algorithm can be used for computing anomalies on large-scale infrastructure point clouds.

#### 1. Generate downsampled point clouds(voxelization) and images(projected from 3D to 2D) 
- Run:
```python
  python multi_view_main.py 
```

#### 2. Evaluate synthetic masonry arch point clouds

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
- (3) Only on varying support movement magnitude(It is not included in this paper for the length limit, though it is a good demonstration for comparing the difference in whether or not new surfaces are added to the synthetic dataset)
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
#### 4. Evaluate real tunnel point clouds 
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

@article{jing2024anomaly,
  title={Anomaly detection of cracks in synthetic masonry arch bridge point clouds using fast point feature histograms and PatchCore},
  author={Jing, Yixiong and Zhong, Jia-Xing and Sheil, Brian and Acikgoz, Sinan},
  journal={Automation in Construction},
  volume={168},
  pages={105766},
  year={2024},
  publisher={Elsevier}
}

@article{jing20253d,
  title={A 3D Multimodal Feature for Infrastructure Anomaly Detection},
  author={Jing, Yixiong and Lin, Wei and Sheil, Brian and Acikgoz, Sinan},
  journal={arXiv preprint arXiv:2502.05779},
  year={2025}
}

## Acknowledge
We used some code from [CPMF](https://github.com/caoyunkang/CPMF) to make comparisons in our work. We would like to thank you for your sharing.

## License
Our work is subjected to MIT License.
