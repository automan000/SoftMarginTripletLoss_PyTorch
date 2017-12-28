import torch
from torch.autograd import Function
from _ext import online_triplet_loss


class OnlineTripletLossFunction(Function):

    def __init__(self, all_triplets, positive_type, negative_type, margin_type, margin, mu):
        self.all_triplets = all_triplets
        self.positive_type = positive_type
        self.negative_type = negative_type
        self.margin_type = margin_type
        self.margin = margin
        self.mu = mu

        # buffers
        self.dist_ = torch.FloatTensor()
        self.agg_ = torch.FloatTensor()
        self.triplets_idxs = torch.IntTensor()
        self.pos_pairs_idxs = torch.IntTensor()
        self.features = None

        self.gpu = None
        self.accuracy = 0.

    def forward(self, features, label):
        if features.is_cuda:
            self.gpu = features.get_device()
            features = features.cpu()
            label = label.cpu()
        else:
            self.gpu = None
        self.features = features

        output = torch.FloatTensor()
        online_triplet_loss.online_triplet_loss_forward(
            features, label, output,
            self.dist_, self.triplets_idxs, self.pos_pairs_idxs,
            self.all_triplets,
            self.positive_type, self.negative_type, self.margin_type,
            self.margin, self.mu
        )

        loss = output[0]
        self.accuracy = output[1]
        loss_tensor = torch.FloatTensor([loss])

        if self.gpu is not None:
            loss_tensor = loss_tensor.cuda(self.gpu)
        return loss_tensor

    def backward(self, grad_top):
        grad_bottom = torch.FloatTensor()
        if grad_top.is_cuda:
            self.gpu = grad_top.get_device()
            grad_top = grad_top.cpu()
        else:
            self.gpu = None

        online_triplet_loss.online_triplet_loss_backward(
            grad_top, self.features, grad_bottom,
            self.agg_, self.dist_,
            self.triplets_idxs, self.pos_pairs_idxs,
            self.mu
        )

        if self.gpu is not None:
            grad_bottom = grad_bottom.cuda(self.gpu)

        return grad_bottom, None


class OnlineTripletLoss(torch.nn.Module):
    SampleMethod_ALL = 0
    SampleMethod_HARD = 1
    SampleMethod_MODERATE = 2

    MarginType_HARD = 0
    MarginType_SOFTMARGIN = 1


    def __init__(self, all_triplets, positive_type, negative_type, margin_type, margin, mu):
        super(OnlineTripletLoss, self).__init__()

        self.all_triplets = all_triplets
        self.positive_type = positive_type
        self.negative_type = negative_type
        self.margin_type = margin_type
        self.margin = margin
        self.mu = mu
        self.accuracy = 0.

    def forward(self, features, label):
        loss_fun = OnlineTripletLossFunction(self.all_triplets, self.positive_type, self.negative_type,
                                             self.margin_type, self.margin, self.mu)
        loss = loss_fun(features, label)
        self.accuracy = loss_fun.accuracy
        return loss
