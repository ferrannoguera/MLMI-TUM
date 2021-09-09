import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, n_classes, with_CE):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.with_CE = with_CE

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True): #[1,2,2,2,2,2,2,2,2]
        if self.with_CE:
        	loss_ce = F.cross_entropy(inputs, target)
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        #print("target: ", target.shape)
        #print("n_classes: ", self.n_classes)
        #print("weight: ", weight)
        if weight is None:
            weight = [1] * self.n_classes
        #print("weight: ", weight)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
            #print("loss of class ", i, ": ", dice * weight[i])
        #print("loss: ", loss / self.n_classes)
        #print("-----------------------")
        if self.with_CE:
        	loss_dice = loss / self.n_classes
        	final_loss = 0.5*loss_ce + 0.5*loss_dice
        else:
            final_loss = loss / self.n_classes
        return final_loss