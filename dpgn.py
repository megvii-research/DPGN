import torch.nn as nn
import torch.nn.functional as F
import torch


class PointSimilarity(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        """
        Point Similarity (see paper 3.2.1) Vp_(l-1) -> Ep_(l)
        :param in_c: number of input channel
        :param base_c: number of base channel
        :param device: the gpu device stores tensors
        :param dropout: dropout rate
        """
        super(PointSimilarity, self).__init__()
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout
        layer_list = []

        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c*2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c*2),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c*2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c, out_channels=1, kernel_size=1)]
        self.point_sim_transform = nn.Sequential(*layer_list)

    def forward(self, vp_last_gen, ep_last_gen, distance_metric):
        """
        Forward method of Point Similarity
        :param vp_last_gen: last generation's node feature of point graph, Vp_(l-1)
        :param ep_last_gen: last generation's edge feature of point graph, Ep_(l-1)
        :param distance_metric: metric for distance
        :return: edge feature of point graph in current generation Ep_(l) (for Point Loss)
                 l2 version of node similarities
        """

        vp_i = vp_last_gen.unsqueeze(2)
        vp_j = torch.transpose(vp_i, 1, 2)
        if distance_metric == 'l2':
            vp_similarity = (vp_i - vp_j)**2
        elif distance_metric == 'l1':
            vp_similarity = torch.abs(vp_i - vp_j)
        trans_similarity = torch.transpose(vp_similarity, 1, 3)
        ep_ij = torch.sigmoid(self.point_sim_transform(trans_similarity))

        # normalization
        diagonal_mask = 1.0 - torch.eye(vp_last_gen.size(1)).unsqueeze(0).repeat(vp_last_gen.size(0), 1, 1).to(ep_last_gen.get_device())
        ep_last_gen *= diagonal_mask
        ep_last_gen_sum = torch.sum(ep_last_gen, -1, True)
        ep_ij = F.normalize(ep_ij.squeeze(1) * ep_last_gen, p=1, dim=-1) * ep_last_gen_sum
        diagonal_reverse_mask = torch.eye(vp_last_gen.size(1)).unsqueeze(0).to(ep_last_gen.get_device())
        ep_ij += (diagonal_reverse_mask + 1e-6)
        ep_ij /= torch.sum(ep_ij, dim=2).unsqueeze(-1)
        node_similarity_l2 = -torch.sum(vp_similarity, 3)
        return ep_ij, node_similarity_l2


class P2DAgg(nn.Module):
    def __init__(self, in_c, out_c):
        """
        P2D Aggregation (see paper 3.2.1) Ep_(l) -> Vd_(l)
        :param in_c: number of input channel for the fc layer
        :param out_c:number of output channel for the fc layer
        """
        super(P2DAgg, self).__init__()
        # add the fc layer
        self.p2d_transform = nn.Sequential(*[nn.Linear(in_features=in_c, out_features=out_c, bias=True),
                                             nn.LeakyReLU()])
        self.out_c = out_c

    def forward(self, point_edge, distribution_node):
        """
        Forward method of P2D Aggregation
        :param point_edge: current generation's edge feature of point graph, Ep_(l)
        :param distribution_node: last generation's node feature of distribution graph, Ed_(l-1)
        :return: current generation's node feature of distribution graph, Vd_(l)
        """
        meta_batch = point_edge.size(0)
        num_sample = point_edge.size(1)
        distribution_node = torch.cat([point_edge[:, :, :self.out_c], distribution_node], dim=2)
        distribution_node = distribution_node.view(meta_batch*num_sample, -1)
        distribution_node = self.p2d_transform(distribution_node)
        distribution_node = distribution_node.view(meta_batch, num_sample, -1)
        return distribution_node


class DistributionSimilarity(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        """
        Distribution Similarity (see paper 3.2.2) Vd_(l) -> Ed_(l)
        :param in_c: number of input channel
        :param base_c: number of base channel
        :param device: the gpu device stores tensors
        :param dropout: dropout rate
        """
        super(DistributionSimilarity, self).__init__()
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout
        layer_list = []

        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c * 2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c * 2),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c * 2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c, out_channels=1, kernel_size=1)]
        self.point_sim_transform = nn.Sequential(*layer_list)

    def forward(self, vd_curr_gen, ed_last_gen, distance_metric):
        """
        Forward method of Distribution Similarity
        :param vd_curr_gen: current generation's node feature of distribution graph, Vd_(l)
        :param ed_last_gen: last generation's edge feature of distribution graph, Ed_(l-1)
        :param distance_metric: metric for distance
        :return: edge feature of point graph in current generation Ep_(l)
        """
        vd_i = vd_curr_gen.unsqueeze(2)
        vd_j = torch.transpose(vd_i, 1, 2)
        if distance_metric == 'l2':
            vd_similarity = (vd_i - vd_j)**2
        elif distance_metric == 'l1':
            vd_similarity = torch.abs(vd_i - vd_j)
        trans_similarity = torch.transpose(vd_similarity, 1, 3)
        ed_ij = torch.sigmoid(self.point_sim_transform(trans_similarity))

        # normalization
        diagonal_mask = 1.0 - torch.eye(vd_curr_gen.size(1)).unsqueeze(0).repeat(vd_curr_gen.size(0), 1, 1).to(ed_last_gen.get_device())
        ed_last_gen *= diagonal_mask
        ed_last_gen_sum = torch.sum(ed_last_gen, -1, True)
        ed_ij = F.normalize(ed_ij.squeeze(1) * ed_last_gen, p=1, dim=-1) * ed_last_gen_sum
        diagonal_reverse_mask = torch.eye(vd_curr_gen.size(1)).unsqueeze(0).to(ed_last_gen.get_device())
        ed_ij += (diagonal_reverse_mask + 1e-6)
        ed_ij /= torch.sum(ed_ij, dim=2).unsqueeze(-1)

        return ed_ij


class D2PAgg(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        """
        D2P Aggregation (see paper 3.2.2) Ed_(l) -> Vp_(l+1)
        :param in_c: number of input channel
        :param base_c: number of base channel
        :param device: the gpu device stores tensors
        :param dropout: dropout rate
        """
        super(D2PAgg, self).__init__()
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout
        layer_list = []
        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c*2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c*2),
                       nn.LeakyReLU()]

        layer_list += [nn.Conv2d(in_channels=self.base_c*2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        self.point_node_transform = nn.Sequential(*layer_list)

    def forward(self, distribution_edge, point_node):
        """
        Forward method of D2P Aggregation
        :param distribution_edge: current generation's edge feature of distribution graph, Ed_(l)
        :param point_node: last generation's node feature of point graph, Vp_(l-1)
        :return: current generation's node feature of point graph, Vp_(l)
        """
        # get size
        meta_batch = point_node.size(0)
        num_sample = point_node.size(1)

        # get eye matrix (batch_size x node_size x node_size)
        diag_mask = 1.0 - torch.eye(num_sample).unsqueeze(0).repeat(meta_batch, 1, 1).to(distribution_edge.get_device())

        # set diagonal as zero and normalize
        edge_feat = F.normalize(distribution_edge * diag_mask, p=1, dim=-1)

        # compute attention and aggregate
        aggr_feat = torch.bmm(edge_feat, point_node)

        node_feat = torch.cat([point_node, aggr_feat], -1).transpose(1, 2)
        # non-linear transform
        node_feat = self.point_node_transform(node_feat.unsqueeze(-1))
        node_feat = node_feat.transpose(1, 2).squeeze(-1)
     
        return node_feat


class DPGN(nn.Module):
    def __init__(self, num_generations, dropout, num_support_sample, num_sample, loss_indicator, point_metric, distribution_metric):
        """
        DPGN model
        :param num_generations: number of total generations
        :param dropout: dropout rate
        :param num_support_sample: number of support sample
        :param num_sample: number of sample
        :param loss_indicator: indicator of what losses are using
        :param point_metric: metric for distance in point graph
        :param distribution_metric: metric for distance in distribution graph
        """
        super(DPGN, self).__init__()
        self.generation = num_generations
        self.dropout = dropout
        self.num_support_sample = num_support_sample
        self.num_sample = num_sample
        self.loss_indicator = loss_indicator
        self.point_metric = point_metric
        self.distribution_metric = distribution_metric
        # node & edge update module can be formulated by yourselves
        P_Sim = PointSimilarity(128, 128, dropout=self.dropout)
        self.add_module('initial_edge', P_Sim)
        for l in range(self.generation):
            D2P = D2PAgg(128*2, 128, dropout=self.dropout if l < self.generation-1 else 0.0)
            P2D = P2DAgg(2*num_support_sample, num_support_sample)
            P_Sim = PointSimilarity(128, 128, dropout=self.dropout if l < self.generation-1 else 0.0)
            D_Sim = DistributionSimilarity(num_support_sample,
                                            num_support_sample,
                                            dropout=self.dropout if l < self.generation-1 else 0.0)
            self.add_module('point2distribution_generation_{}'.format(l), P2D)
            self.add_module('distribution2point_generation_{}'.format(l), D2P)
            self.add_module('point_sim_generation_{}'.format(l), P_Sim)
            self.add_module('distribution_sim_generation_{}'.format(l), D_Sim)

    def forward(self, middle_node, point_node, distribution_node, distribution_edge, point_edge):
        """
        Forward method of DPGN
        :param middle_node: feature extracted from second last layer of Embedding Network
        :param point_node: feature extracted from last layer of Embedding Network
        :param distribution_node: initialized nodes of distribution graph
        :param distribution_edge: initialized edges of distribution graph
        :param point_edge: initialized edge of point graph
        :return: classification result
                 instance_similarity
                 distribution_similarity
        """
        point_similarities = []
        distribution_similarities = []
        node_similarities_l2 = []
        point_edge, _ = self._modules['initial_edge'](middle_node, point_edge, self.point_metric)
        for l in range(self.generation):
            point_edge, node_similarity_l2 = self._modules['point_sim_generation_{}'.format(l)](point_node, point_edge, self.point_metric)
            distribution_node = self._modules['point2distribution_generation_{}'.format(l)](point_edge, distribution_node)
            distribution_edge = self._modules['distribution_sim_generation_{}'.format(l)](distribution_node, distribution_edge, self.distribution_metric)
            point_node = self._modules['distribution2point_generation_{}'.format(l)](distribution_edge, point_node)
            point_similarities.append(point_edge * self.loss_indicator[0])
            node_similarities_l2.append(node_similarity_l2 * self.loss_indicator[1])
            distribution_similarities.append(distribution_edge * self.loss_indicator[2])
        return point_similarities, node_similarities_l2, distribution_similarities




