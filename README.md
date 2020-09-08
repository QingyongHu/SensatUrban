# Towards Semantic Segmentation of Urban-Scale 3D Point Clouds: A Dataset, Benchmarks and Challenges

This is the official repository of the **SensatUrban** dataset. For technical details, please refer to:

**Towards Semantic Segmentation of Urban-Scale 3D Point Clouds: A Dataset, Benchmarks and Challenges** <br />
[Qingyong Hu](https://qingyonghu.github.io/), [Bo Yang*](https://yang7879.github.io/), [Sheikh Khalid](https://uk.linkedin.com/in/fakharkhalid), 
[Wen Xiao](https://www.ncl.ac.uk/engineering/staff/profile/wenxiao.html), [Niki Trigoni](https://www.cs.ox.ac.uk/people/niki.trigoni/), [Andrew Markham](https://www.cs.ox.ac.uk/people/andrew.markham/). <br />
**[[Paper](http://arxiv.org/abs/2009.03137)] [[Video](https://www.youtube.com/watch?v=IG0tTdqB3L8)] [[Project page](https://github.com/QingyongHu/SensatUrban)] [Download]** <br />

### (1) Dataset

#### 1.1 Overview

This dataset is an urban-scale photogrammetric point cloud dataset with nearly three billion richly annotated points, 
which is five times the number of labeled points than the existing largest point cloud dataset. 
Our dataset consists of large areas from two UK cities, covering about 6 km^2 of the city landscape. 
In the dataset, each 3D point is labeled as one of 13 semantic classes, such as *ground*, *vegetation*, 
*car*, *etc.*. 

<p align="center"> <img src="figs/Fig2.png" width="70%"> </p>
<p align="center"> <img src="figs/Fig1.png" width="100%"> </p>

#### 1.2 Data collection

The 3D point clouds are generated from high-quality aerial images captured by a 
professional-grade UAV mapping system. In order to fully and evenly cover the survey area, 
all flight paths are pre-planned in a grid fashion and automated by the flight control system (e-Motion).

<p align="center"> <img src="figs/Fig3.png" width="70%"> </p>

#### 1.3 Semantic Annotation

<p align="center"> <img src="figs/Fig4.png" width="100%"> </p>

- Ground: including impervious surfaces, grass, terrain
- Vegetation: including trees, shrubs, hedges, bushes
- Building: including commercial / residential buildings
- Wall: including fence, highway barriers, walls
- Bridge: road bridges
- Parking: parking lots
- Rail: railroad tracks
- Traffic Road: including main streets, highways
- Street Furniture: including benches, poles, lights
- Car: including cars, trucks, HGVs
- Footpath: including walkway, alley
- Bike: bikes / bicyclists
- Water: rivers / water canals


#### 1.4 Statistics
<p align="center"> <img src="figs/Fig5.png" width="100%"> </p>


### (2) Benchmark
We extensively evaluate the performance of state-of-the-art algorithms on our dataset 
and provide a comprehensive analysis of the results. In particular, we identify several key challenges 
towards urban-scale point cloud understanding. 

<p align="center"> <img src="figs/Fig6.png" width="100%"> </p>


### (3) Demo

<p align="center"> <a href="https://youtu.be/IG0tTdqB3L8"><img src="./figs/3DV_demo_cover.png" width="50%"></a> </p>

### Citation
If you find our work useful in your research, please consider citing:

	@article{hu2019randla,
	  title={Towards Semantic Segmentation of Urban-Scale 3D Point Clouds: A Dataset, Benchmarks and Challenges},
	  author={Hu, Qingyong and Yang, Bo and Khalid, Sheikh and Xiao, Wen and Trigoni, Niki and Markham, Andrew},
	  journal={arXiv preprint arXiv:2009.03137},
	  year={2020}
	}




