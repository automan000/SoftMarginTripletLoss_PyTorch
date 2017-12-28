import numpy as np
import random
import torch
from torch.autograd.variable import Variable

from online_triplet_loss.online_triplet_loss import OnlineTripletLoss as LossFunction


def produce_labels(initial_length=100):
    id_num = []
    while initial_length > 0:
        single = random.randint(1, 10)
        if single <= initial_length:
            id_num.append(single)
            initial_length -= single
    random.shuffle(id_num)
    labels = []
    for index, num in enumerate(id_num):
        labels.extend((index * np.ones(shape=num)).tolist())
    random.shuffle(labels)
    labels = np.asarray(labels).astype(np.int32)
    return labels


def run_check():
    triplet_loss_fun = LossFunction(all_triplets=False,
                                    positive_type=LossFunction.SampleMethod_ALL,
                                    negative_type=LossFunction.SampleMethod_ALL,
                                    margin_type=LossFunction.MarginType_SOFTMARGIN,
                                    margin=0.3, mu=1.)

    input = Variable(torch.rand(100, 256)).cuda()
    target = Variable(torch.IntTensor(produce_labels(100))).cuda()

    res = torch.autograd.gradcheck(triplet_loss_fun, (input, target), raise_exception=True)
    print(res)


if __name__ == '__main__':
    run_check()
