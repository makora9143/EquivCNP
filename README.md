# LieConv Conditional Neural Process



## 1D regression Task

- `learning_rate=1e-3`
- `batch_size=16`
- `Adam`



### PointConv CNP

- `WeightNet`: use batch normalization
- `PointConv`: `use mean=True`
- `sampling_fraction=1.0`
- `nbhd = 5`: as well as `kernel_size` of Conv1d


### LieConv CNP

- `WeightNet`: use batch normalization
- `LieConv` : `use mean=True`
- `sampling_fraction=1.0`
- `fill = 5 / 64`
- `nbhd = 5`


## MNIST


### PointConv

- `WeightNet`: use batch normalization
- `PointConv`: `use mean=True`
- `sampling_fraction=1.0`
- `nbhd = 9`: as well as `kernel_size` of Conv1d
