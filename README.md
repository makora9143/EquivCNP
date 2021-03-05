# Group Equivariant Conditional Neural Process


## Install

Recommend: Use pytorch Docker images v1.6

Then,
```
$ pip install -r requirements.txt
```


```
$ python main1d.py batch_size=16 dataset=rbf learning_rate=1e-3 model=gcnp num=10
$ python main1d.py batch_size=16 dataset=matern learning_rate=1e-3 model=gcnp num=10
$ python main1d.py batch_size=16 dataset=periodic learning_rate=1e-3 model=gcnp num=5
```

```
$ python main2d.py -m batch_size=4 dataset=clockdigit epochs=100 group=T2,SO2,RxSO2,SE2 learning_rate=5e-4 model=liecnp
```


## BibTeX

```
@inproceedings{kawano2021group,
  author    = {Makoto Kawano and
               Wataru Kumagai and
               Akiyoshi Sannai and
               Yusuke Iwasawa and
               Yutaka Matsuo},
  title     = {Group Equivariant Conditional Neural Processes},
  booktitle = {9th International Conference on Learning Representations, {ICLR} 2021,
               Virtual only, May 3-7, 2021},
  publisher = {OpenReview.net},
  year      = {2021},
  url       = {https://openreview.net/forum?id=e8W-hsu_q5},
}
```