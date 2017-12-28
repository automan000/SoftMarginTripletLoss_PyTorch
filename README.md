# SoftMarginTripletLoss_Pytorch

This repository is based on previous works from [Part_ReID](https://github.com/zlmzju/part_reid).

Please refer to [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/pdf/1703.07737) and [Deeply-Learned Part-Aligned Representations for Person Re-Identification](https://arxiv.org/pdf/1707.07256) for more details.


# Formulation

<a href="https://www.codecogs.com/eqnedit.php?latex=loss&space;=&space;mu&space;\times&space;loss_{triplet}&space;&plus;&space;(1-mu)&space;\times&space;loss_{intraclass}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?loss&space;=&space;mu&space;\times&space;loss_{triplet}&space;&plus;&space;(1-mu)&space;\times&space;loss_{intraclass}" title="loss = mu \times loss_{triplet} + (1-mu) \times loss_{intraclass}" /></a>


# install

```
cd online_triplet_loss
sh ./mask.sh
```


# Usage

## definition
```python
from online_triplet_loss.online_triplet_loss import OnlineTripletLoss as LossFunction

triplet_loss_fun = LossFunction(all_triplets=False,
                                positive_type=LossFunction.SampleMethod_ALL,
                                negative_type=LossFunction.SampleMethod_ALL,
                                margin_type=LossFunction.MarginType_SOFTMARGIN,
                                margin=0.3, mu=1.)
```

## Foward
```python
loss = triplet_loss_fun(features, labels)
accuracy = triplet_loss_fun.accuracy
```


## optional arguments

#### sample strategy:
+ SampleMethod_ALL
+ SampleMethod_HARD
+ SampleMethod_MODERATE

#### margin type:

+ hard margin
+ soft margin

#### mu
+ This parameter is adopted for keep a balance between the intra-class loss for postive pairs and the triplet distance.
+ mu=1. is highly recommend, which means that we ignore the intra-class loss.
