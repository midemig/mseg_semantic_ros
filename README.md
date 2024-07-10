# mseg_semantic_ros
ROS 2 implementation of mseg-semantic.


## Instalation

Follow install instructions for:

[https://github.com/mseg-dataset/mseg-semantic](https://github.com/mseg-dataset/mseg-semantic) and [https://github.com/mseg-dataset/mseg-api](https://github.com/mseg-dataset/mseg-api)

or

```bash
cd ~/
git clone https://github.com/mseg-dataset/mseg-semantic
pip install -e mseg-semantic
git clone https://github.com/mseg-dataset/mseg-api
pip install -e mseg-api
```

Colcon build

```bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --symlink-install
```

Download Weights from [here](https://github.com/mseg-dataset/mseg-semantic) (MSeg Pre-trained Models)

Set model name and model path in `/cfg/default_config_360_ms.yaml` file




## Run

```bash
ros2 run mseg_semantic_ros sem_seg_node
```

### Input topic

- /camera/color/image_raw

### Output topic

- /camera/semseg/image