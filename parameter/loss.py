

import torch
import torch.nn as nn
import torch.nn.functional as F

 
class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
        
    def forward(self, p, q):
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        loss = F.kl_div(q.log(), p, reduction='batchmean')
        return loss


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
#改写 https://zhuanlan.zhihu.com/p/295512971 tensorflow版为torch
class BatchAllTripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(BatchAllTripletLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-5

    def _pairwise_distance(self, embeddings,  squared=False):
        '''
            计算两两embedding的距离
            ------------------------------------------
            Args：
                embedding: 特征向量， 大小（batch_size, vector_size）
                squared:   是否距离的平方，即欧式距离
            Returns：
                distances: 两两embeddings的距离矩阵，大小 （batch_size, batch_size）
        '''    
        # 矩阵相乘,得到（batch_size, batch_size），因为计算欧式距离|a-b|^2 = a^2 -2ab + b^2, 
        # 其中 ab 可以用矩阵乘表示
        dot_product =  torch.mm(embeddings, embeddings.T)   
        # dot_product对角线部分就是 每个embedding的平方
        square_norm = torch.diag(dot_product)
        # |a-b|^2 = a^2 - 2ab + b^2
        # tf.expand_dims(square_norm, axis=1)是（batch_size, 1）大小的矩阵，减去 （batch_size, batch_size）大小的矩阵，相当于每一列操作
        distances = square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)
        distances = torch.maximum(distances, torch.tensor(0.0))  # 小于0的距离置为0
        if not squared:          # 如果不平方，就开根号，但是注意有0元素，所以0的位置加上 1e*-16
            mask = torch.eq(distances, 0.0).int()
            distances = distances + mask * self.eps #这里加过eps了，后面还需要吗
            distances = torch.sqrt(distances)  + self.eps

            # 当x为0时在计算标准差时正向传播需要计算sqrt(x)
            # 而反向传播时需要计算sqrt(x)的微分1/(2*sqrt(x))此时需要确保x != 0

            distances = distances * (1.0 - mask)    # 0的部分仍然置为0
        return distances

    def _get_triplet_mask(self, labels, rank):
        '''
            得到一个3D的mask [a, p, n], 对应triplet（a, p, n）是valid的位置是True
            ----------------------------------
            Args:
                labels: 对应训练数据的labels, shape = (batch_size,)
            Returns:
                mask: 3D,shape = (batch_size, batch_size, batch_size)
        '''
        # 初始化一个二维矩阵，坐标(i, j)不相等置为1，得到indices_not_equal
        indices_equal = torch.eye(labels.shape[0]).bool().to(rank)
        indices_not_equal = (~indices_equal).int()
        # 因为最后得到一个3D的mask矩阵(i, j, k)，增加一个维度，则 i_not_equal_j 在第三个维度增加一个即，(batch_size, batch_size, 1), 其他同理
        i_not_equal_j = indices_not_equal.unsqueeze(2) 
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)
        # 想得到i!=j!=k, 三个不等取and即可, 最后可以得到当下标（i, j, k）不相等时才取True
        #不知道我改成这样对不对
        distinct_indices = (( ((i_not_equal_j + i_not_equal_k)==2).int() + j_not_equal_k)==2).int()
        # 同样根据labels得到对应i=j(anchor和正样本要同类), i!=k（anchor与负样本要异类）
        label_equal =(labels.unsqueeze(0) == labels.unsqueeze(1)).int()
        i_equal_j = label_equal.unsqueeze(2) 
        i_equal_k = label_equal.unsqueeze(1)
        valid_labels = ( i_equal_j +  (1 - i_equal_k) ==2).int()
        # mask即为满足上面两个约束，所以两个3D取and
        mask = ((distinct_indices + valid_labels)==2).int()
        return mask

    def batch_all_triplet_loss(self, labels, embeddings, rank, margin=None, squared=False):
        '''
            triplet loss of a batch
            -------------------------------
            Args:
                labels:     标签数据，shape = （batch_size,）
                embeddings: 提取的特征向量， shape = (batch_size, vector_size)
                margin:     margin大小， scalar
            Returns:
                triplet_loss: scalar, 一个batch的损失值
                fraction_postive_triplets : valid的triplets占的比例
        '''
        if margin==None:
            margin=self.margin
        
        # 得到每两两embeddings的距离，然后增加一个维度，一维需要得到（batch_size, batch_size, batch_size）大小的3D矩阵
        # 然后再点乘上valid 的 mask即可
        
        #1、计算所有3元组的loss
        pairwise_dis = self._pairwise_distance(embeddings, squared=squared)
        anchor_positive_dist = pairwise_dis.unsqueeze(2)
        assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
        anchor_negative_dist = pairwise_dis.unsqueeze(1)
        assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
        # 2、用mask去掉不应该计算的loss
        mask = self._get_triplet_mask(labels, rank)
        mask = mask.float()
        triplet_loss = torch.multiply(mask, triplet_loss)
        triplet_loss = torch.maximum(triplet_loss, torch.tensor(0.0))
        # 计算valid的triplet的个数，然后对所有的triplet loss求平均
        valid_triplets = (triplet_loss> 1e-5).float()
        num_positive_triplets = torch.sum(valid_triplets)
        num_valid_triplets = torch.sum(mask)
        fraction_postive_triplets = num_positive_triplets / (num_valid_triplets + 1e-5)
        triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-5)
        return triplet_loss, fraction_postive_triplets

    def forward(self, embeddings, labels, margin=None, squared=False):
        if margin==None:
            margin=self.margin
        return self.batch_all_triplet_loss(labels, embeddings,margin, squared=False,)


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# 通过log_prob做的异类特征推远
# log_prob = -torch.log(exp_dot_tempered / (
#     torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
# 对比loss：用于单模态内，同类特征拉近
# https://github.com/GuillaumeErhard/Supervised_contrastive_loss_pytorch/blob/main/loss/spc.py
import torch
import torch.nn as nn
from math import log
class SupervisedContrastiveLoss_myrevised(nn.Module):
    def __init__(self, temperature=0.07):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362
        :param temperature: int
        """
        super(SupervisedContrastiveLoss_myrevised, self).__init__()
        self.temperature = temperature

    def forward(self, projections, targets, rank):
        """
        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        # device = torch.device(
        #     "cuda:"+cudaNO) if projections.is_cuda else torch.device("cpu")

        # print(projections)  # trans的输出已经layernorm过了，但没有每个样本的模norm，所以乘起来很大

        dot_product_tempered = torch.mm(
            projections, projections.T) / self.temperature

        # print(dot_product_tempered)  # 600 700因为维度多再加上temperature的作用
        # # tensor([[651.5817, 590.9725],
        # #         [590.9725, 637.5401]], device='cuda:0')

        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        # 这里数值稳定性保持的不好，应该提前 对每个样本的特征做模归一化  即z-score后除上特征维度
        exp_dot_tempered = (
            torch.exp(dot_product_tempered -
                      torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        # # torch.max返回最大值[0] 和其位置[1]
        # print(torch.max(dot_product_tempered, dim=1, keepdim=True)[0])
        # # tensor([[651.5817],
        # #         [637.5401]], device='cuda:0')

        # # 上面dot_product_tempered太大，导致这里相似度矩阵直接1和0了        可能就是这里导致后面log出现nan
        # print(exp_dot_tempered)
        # # tensor([[1.0000e+00, 1.0000e-05],
        # #         [1.0000e-05, 1.0000e+00]], device='cuda:0')
        # 对 特征 除掉 模的平方后，这里几乎都是0.99了

        # return

        mask_similar_class = (targets.unsqueeze(1).repeat(
            1, targets.shape[0]) == targets).to(rank)  # label构成的正负样本mask
        # print(mask_similar_class)

        mask_anchor_out = (
            1 - torch.eye(exp_dot_tempered.shape[0])).to(rank)  # 对角线不参与正样本计算
        # print(mask_anchor_out)

        mask_combined = mask_similar_class * mask_anchor_out  # 正负样本mask
        # print(mask_combined)

        cardinality_per_samples = torch.sum(mask_combined, dim=1)  # 我估计是正样本数吧
        # print(cardinality_per_samples)
        # [1., 2., 2., 3., 3., 1., 2., 3., 3., 3., 1., 1., 3., 0., 3., 3.]
        # 为什么出现nan，因为有的样本就没有正样本啊 后面一除0，就是nan了
        # 加大batchsize可以，但是注意末尾的batch大小很小，就会出现nan，还是检测一下吧

        my_delete_zeros = cardinality_per_samples.bool()  # 我改的就只有这里

        log_prob = -torch.log(exp_dot_tempered / (
            torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))

        supervised_contrastive_loss_per_sample = torch.sum(
            log_prob * mask_combined, dim=1)[my_delete_zeros] / cardinality_per_samples[my_delete_zeros]  # 我改的就只有这里my_delete_zeros
        supervised_contrastive_loss = torch.mean(
            supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# 对比loss：用于拉齐两种模态的特征
# from infonce_copied import InfoNCE
import torch
import torch.nn.functional as F
from torch import nn
__all__ = ['InfoNCE', 'info_nce']
def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError(
                "<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError(
                "<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError(
            '<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError(
                "If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError(
            'Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError(
                'Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(
        query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(
            len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)
# logits即概率
# @对于tensor就是矩阵乘法
def transpose(x):
    return x.transpose(-2, -1)
# 选择2个维度互换位置，因此参数顺序无所谓，x.transpose(-1, -2)也一样
def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]
# 每个样本内归一化

class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.
    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.
    Returns:
         Value of the InfoNCE Loss.
     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)

