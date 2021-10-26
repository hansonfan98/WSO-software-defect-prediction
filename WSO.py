#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/8/2 10:19
# @Author : hq_fan
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors

# 修改后面的difficult_factor变成safe_factor， 利用relative_score来实现对其的表示


class WSO:
    def __init__(self, a, b, c, d, e, g, h, i):
        self.x_train = a  # 训练集
        self.y_train = b  # 训练集对应的标签
        self.min_class = c  # 少数类类别
        self.maj_class = d  # 多数类类别
        self.num_create = e  # 需要生成的样本个数

        self.k1 = g  # 计算hub_aware时的k值
        self.k2 = h  # 计算生成时候的k值
        self.w = i  # 这个值用来对safe_factor因子加权

        self._divide_region()

    def _divide_region(self):
        x_train = self.x_train
        y_train = self.y_train
        self.index_1 = np.where(y_train == self.min_class)  # 少数类样本索引
        self.index_2 = np.where(y_train == self.maj_class)  # 多数类样本索引

        self.y_train_p = y_train[self.index_1]
        self.y_train_n = y_train[self.index_2]  # 标签中的负样本对应的标签

        self.positive_samples = x_train[self.index_1]

        self.negative_samples = x_train[self.index_2]

        self.imb_ratio = len(self.negative_samples) / len(self.positive_samples)
        self.num_create = len(self.negative_samples) - len(self.positive_samples)

    # 计算relative_score
    def hub_aware(self):
        min_index = self.index_1

        min_class = self.min_class
        x_train = self.x_train
        y_train = self.y_train
        pos = self.positive_samples
        k_count = self.k1  # 确定hub_aware时的k值

        # fitting the model
        n_neigh = min([len(pos), k_count])
        nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
        nn.fit(x_train)
        dist, ind = nn.kneighbors(x_train)

        # 对整个训练集的样本进行寻找其坏的邻居和好的邻居
        lis = [[] for i in range(len(x_train))]
        for element_1 in range(len(ind)):
            ind_1 = ind[element_1]
            ind_new = ind_1[1:]
            label_ = y_train[element_1]
            if label_ == min_class:  # 如果种子为少数类
                for i in ind_new:
                    if y_train[i] == min_class:
                        lis[i].append('g')
                    else:
                        lis[i].append('b')
            else:
                for i in ind_new:
                    if y_train[i] == min_class:
                        lis[i].append('b')
                    else:
                        lis[i].append('g')
        count = [[] for i in range(len(x_train))]
        for element_2 in range(len(lis)):
            g = lis[element_2].count('g')
            b = lis[element_2].count('b')
            sum_ = g + b + 1
            r_score = g / sum_
            count[element_2].append(r_score)

        count = np.array(count).flatten()
        pos_score = count[min_index]  # 少数类样本对应的好的计数占比， 越大的话表明样本越安全
        relative_score = pos_score  # 保存select_samples对应的relative_score

        return relative_score

    # 选择对应辅助样本的权重
    def select_weight(self):
        relative_score = self.hub_aware()
        relative_score = np.array(relative_score)

        k_weight = self.k2    # 确定选择权重时候的k值
        w1 = self.w   # 对safe_factor进行加权

        pos = self.positive_samples
        neg = self.negative_samples

        # fitting the model
        n_neigh = min([len(pos), k_weight])
        nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
        nn.fit(pos)
        dist, ind = nn.kneighbors(pos)  # 只找到对应的select_samples的同类近邻

        # 为每个样本创建一个二维列表，存储safe_factor
        safe_factor = [[] for _ in range(len(pos))]

        # 为每个列表创建一个二维列表，存储generalization_factor
        generalization_factor = [[] for _ in range(len(pos))]

        # 对于每个少数类样本，找到其泛化区域并选择每个泛化区域的权重，包括多数类个数以及距离影响因素
        for element_3 in range(len(ind)):
            distance__ = dist[element_3]
            distance_ = distance__[1:n_neigh]  # 近邻样本对应的距离
            max_dist = distance_[-1]  # 最远的近邻样本距离

            # 找到hypersphere 内部的多数类样本
            n_neigh_neg = len(neg)
            nn_neg = NearestNeighbors(n_neighbors=n_neigh_neg, n_jobs=1)
            nn_neg.fit(neg)
            dist_neg, ind_neg = nn_neg.kneighbors([pos[element_3]])

            # 转成一维数组
            dist_i = [i for item in dist_neg for i in item]
            ind_i = [i for item in ind_neg for i in item]

            list_neg = list()  # 存储的是最远近邻以内的所有多数类样本
            for element_4 in range(len(dist_i)):
                if dist_i[element_4] <= max_dist:
                    sam = neg[ind_i[element_4]]
                    list_neg.append(sam)
            array_neg = np.array(list_neg)  # 最远的同类样本内部的所有多数类样本

            # 计算relative_score
            neighbor_ind = ind[element_3]
            neighbor_ind = neighbor_ind[1:n_neigh]  # 近邻样本的索引

            # 如果最大的泛化区域中无多数类样本,safe_factor计算的是relative_score
            if len(array_neg) == 0:
                for i in range(n_neigh-1):
                    generalization_factor[element_3].append(math.exp(0))
                    # 对应的relative_score
                    relative_score_i = relative_score[neighbor_ind[i]]  # 第i个样本的相关分数
                    safe_factor[element_3].append(np.exp(relative_score_i))
            # 最大的泛化区域中有多数类样本
            else:
                # 为候选样本找到其特定领域内的多数类样本
                nbr_samples = pos[neighbor_ind]  # 这个候选样本对应的近邻同类样本

                area_ = distance_  # 这个样本对应的近邻的同类样本距离
                area = area_ / np.sum(area_)  # 种子样本对应的距离,并归一化
                for element_5 in range(len(nbr_samples)):  # 注意这里是对于这个种子样本的每个近邻样本
                    seed_samples = pos[element_3]  # 种子样本
                    assist_samples = nbr_samples[element_5]  # 辅助样本

                    attr_min = [[] for _ in range(len(seed_samples))]
                    attr_max = [[] for _ in range(len(seed_samples))]
                    # 种子样本与辅助样本的属性最大值和最小值
                    for attr in range(len(seed_samples)):
                        min_attr = min(seed_samples[attr], assist_samples[attr])
                        max_attr = max(seed_samples[attr], assist_samples[attr])
                        attr_min[attr].append(min_attr)
                        attr_max[attr].append(max_attr)

                    # 找到array_neg中符合条件的样本
                    select_neg = list()  # 存储这个合成区域中的多数类样本
                    for sam in array_neg:
                        evalu = list()
                        for att in range(len(sam)):
                            if attr_min[att] <= sam[att] <= attr_max[att]:
                                evalu.append(1)
                        if len(evalu) == len(sam):
                            select_neg.append(sam)

                    if len(select_neg) == 0:  # 也就是说这个合成区域是没有多数类样本的
                        density_ratio = 0
                        v1 = 1
                        generalization_ = 1 / (np.exp(density_ratio + v1))
                        generalization_factor[element_3].append(generalization_)

                        relative_score_i = relative_score[neighbor_ind[element_5]]  # 第element_5个样本的相关分数
                        safe_factor[element_3].append(np.exp(relative_score_i))

                    elif len(select_neg) == 1:  # 也就是说这个合成区域只有一个多数类样本
                        # 找到array_neg中符合条件的样本
                        num_factor = len(select_neg) / len(array_neg)
                        distance_factor = area[element_5]
                        density_ratio = num_factor / (distance_factor + 1)  # density_ratio  DR
                        v1 = 0
                        generalization_ = 1 / (np.exp(density_ratio + v1))
                        generalization_factor[element_3].append(generalization_)

                        relative_score_i = relative_score[neighbor_ind[element_5]]  # 第element_5个样本的相关分数
                        safe_factor[element_3].append(np.exp(relative_score_i))
                    else:
                        # 找到array_neg中符合条件的样本
                        num_factor = len(select_neg) / len(array_neg)
                        distance_factor = area[element_5]
                        density_ratio = num_factor / (distance_factor + 1)  # density_ratio  DR

                        # print(select_neg)

                        # 计算V1， volume of negative region
                        # 找到select_neg的样本区域
                        attr1_min = [[] for _ in range(len(seed_samples))]
                        attr1_max = [[] for _ in range(len(seed_samples))]
                        for attr1 in range(len(seed_samples)):
                            attr_list = list()
                            for i in select_neg:
                                attr_list.append(i[attr1])

                            max_attr1 = max(attr_list)
                            min_attr1 = min(attr_list)
                            attr1_min[attr1].append(min_attr1)
                            attr1_max[attr1].append(max_attr1)
                        attr_min = np.array(attr_min).flatten()
                        attr_max = np.array(attr_max).flatten()
                        attr1_min = np.array(attr1_min).flatten()
                        attr1_max = np.array(attr1_max).flatten()

                        v1 = 1
                        for i in range(len(attr_max)):
                            range_s = abs(attr_max[i] - attr_min[i])
                            v_neg = abs(attr1_max[i] - attr1_min[i])
                            if range_s == 0 or v_neg == 0:  # 如果两个特征值为零的话
                                v1 = v1 * 1
                            else:
                                v1 = v1 * (v_neg / range_s)
                        generalization_ = 1 / (np.exp(density_ratio + v1))
                        generalization_factor[element_3].append(generalization_)

                        # 计算safe_factor
                        relative_score_i = relative_score[neighbor_ind[element_5]]  # 第element_5个样本的相关分数
                        safe_factor[element_3].append(np.exp(relative_score_i))
        factor = (w1 * np.array(generalization_factor)) + ((1-w1) * np.array(safe_factor))
        return factor

    # 对少数类关键样本进行采样
    def pos_sampling(self):

        pos = self.positive_samples
        neg = self.negative_samples
        label_neg = self.y_train_n
        num_need_create = self.num_create
        min_class = self.min_class
        y_train_min = self.y_train_p

        k_weight = self.k2  # 近邻时候的k值
        attr = pos.shape[1]
        factor = self.select_weight()

        new_factor = list()
        # 归一化
        for i in range(len(factor)):
            factor_ = factor[i] / np.sum(factor[i])
            new_factor.append(factor_)
        new_factor = np.array(new_factor)

        # fitting the model
        n_neigh = min([len(pos), k_weight])
        if n_neigh == 1:
            n_neigh = n_neigh + 1
        else:
            n_neigh = n_neigh
        nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
        nn.fit(pos)
        dist, ind = nn.kneighbors(pos)

        # generating samples
        new = list()
        base_indices = np.random.choice(list(range(len(pos))), num_need_create)  # num_create_final
        # 从大小为e的list(range(len(a))生成非均匀随机样本,权重为weight, p = weight

        for j in range(len(base_indices)):
            final_factor = new_factor[base_indices[j]]

            # print(final_factor)

            neighbor_indices = np.random.choice(list(range(1, n_neigh)), p=final_factor)
            # 从大小为e的list(range(1, n_neigh)生成均匀随机样本
            # print(neighbor_indices)

            x_base = pos[base_indices[j]]
            x_neighbor = pos[ind[base_indices[j], neighbor_indices]]  # 找到的是ind中第base_indices个样本的
            # 第neighbor_indices索引对应的

            giff = np.random.rand(attr)

            samples = x_base + np.multiply((x_neighbor - x_base), giff)

            new.append(samples)

        x_final = np.vstack((pos, np.array(new)))

        if num_need_create == 0:
            x_label_final = y_train_min
        else:
            x_label_final = np.hstack((y_train_min, np.hstack([min_class] * num_need_create)))

        x_train_final = np.vstack((x_final, neg))
        x_train_label_final = np.hstack((x_label_final, label_neg))

        return x_train_final, x_train_label_final
