# SoftMarginTripletLoss_Pytorch

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


### optional arguments

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

## Foward
```python
loss, accuracy = triplet_loss_fun(features, labels)
```