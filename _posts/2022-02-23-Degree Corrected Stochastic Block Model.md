---
layout: post
title: Degree Corrected Stochastic Block Model
tags: [Algorithm]
date: 2022-02-23 18:46
---

本文基于论文2010 - Stochastic blockmodels and community structure in networks. 作为DC-SBM的学习笔记。

## Traditional SBM

经典的随机块模型定义了一个矩阵，记录着组间连边的概率。这篇文章考虑一般情况，假设节点之间存在多条连边的情况（同时存在自连边），此时这个概率矩阵变为连边数的期望矩阵$\omega$。$\omega_{rs}$表示组$r$和$s$之间连边数的期望。但是对角线上连边数的期望为$\omega_{rr}/2$。

设网络为$G$。网络邻接矩阵为$A$，$A_{ij}$为节点$i$，$j$之间的连边数，但是$A_{ii}$为节点$i$的自连边的两倍。$g$为节点的组，$g_i=r$表示节点$i$的组是$r$。

### 基于SBM的社团分类，极大似然法推导
设节点间的边对数是相互独立的泊松分布，且期望为$\omega_{g_ig_j}$。为了在已知网络的邻接矩阵的情况下进行社团检测，我们需要给出在已知参数$\omega, g$的情况下，生成该网络的概率$P(G|\omega, g)$， 然后最大化这个概率即可找到在SBM假设下的社团检测结果$g$（极大似然法）。这个概率为：

$$\begin{aligned}P(G|\omega, g)&=\prod_{i<j}\frac{(\omega_{g_ig_j})^{A_{ij}}}{A_{ij}!}e^{-\omega_{g_ig_j}}\times\prod_{i}\frac{(\frac{1}{2}\omega_{g_ig_i})^{A_{ii}/2}}{(A_{ii}/2)!}e^{-\frac{1}{2}\omega_{g_ig_i}}\\&=\frac{1}{\prod_{i<j}A_{ij}!\prod_{i}2^{A_{ii}/2}(A_{ii}/2)!}[\prod_{i<j}(\omega_{g_ig_j})^{A_{ij}}e^{-\omega_{g_ig_j}}\prod_{i}(\omega_{g_ig_i})^{A_{ii}/2}e^{-\frac{1}{2}\omega_{g_ig_i}}]\\&=\frac{1}{\prod_{i<j}A_{ij}!\prod_{i}2^{A_{ii}/2}(A_{ii}/2)!}[\prod_{r,s}(\omega_{rs})^{\frac{1}{2}(edges\ between\ r,s)}e^{-\frac{1}{2}n_rn_sw_{rs}}]\end{aligned}$$

设组$r$, $s$之间的边数为$m_{rs}$

$m_{rs}=\sum_{ij}A_{ij}\delta_{g_i,r}\delta_{g_j, s}$

同时删掉$A_{ij}$和其他多余的部分，在对这个概率取对数（极大似然法常用的方法），最终得到优化的目标为：

$$logP(G|\omega, g)=\sum_{rs}(m_{rs}log\omega_{rs}-n_rn_s\omega_{rs})\tag{1}$$

对其求导求极值：

$$\begin{aligned}f_{\omega}^{`}=\sum_{rs}\frac{m_{rs}}{\omega_{rs}}-n_rn_s&=0\\\widehat{\omega_{rs}}&=\frac{m_{rs}}{n_rn_s}\end{aligned}$$

将$\widehat{\omega_{rs}}$代入优化目标$(1)$, 并删除最后一项（其求和为网络边数得的2倍，是常数项）得：

$$\mathfrak{L}(G|g)=\sum_{rs}m_{rs}log\frac{m_{rs}}{n_rn_s} \tag{2}$$

这个式子中的$m_{rs}, n_r, n_s$都与$g$有关，最大化这个式子得到的就是社团检测的结果。

### 信息论解释

设$m$为网络的边数，对式$(2)$进行变换并忽略常数项得到：

$$\mathfrak{L}(G|g)=\sum_{rs}\frac{m_{rs}}{2m}log\frac{m_{rs}/2m}{n_rn_s/n^2}$$

其中$m_{rs}/2m$是一个概率。假设从网络中随机抽一条边，则这条边一端在$r$内，另一端在$s$内的概率即为$p_K(r,s) = m_{rs}/2m$。当网络分组相同，但连边不同时，另一项$n_rn_s/2m$是这个概率的期望值，记为$p_1(r,s)$。该式变为

$$\mathfrak{L}(G|g)=\sum_{rs}p_K(r,s)log\frac{p_K(r,s)}{p_1(r,s)}$$

这是$p_K$与$p_1$之间的KL散度。表示着$p_K$距离$p_1$有多远。所以最大化这个式子意味着：最有可能的分组是在那些没有分组结构的网络的基础上需要最多信息来描述的分组。

## Degree Corrected SBM
度修正的随机块模型在经典的SBM的基础上只修改了节点间连边的期望。在SBM中，节点间连边数的期望为$\omega_{g_ig_j}$，而在DC-SBM中，这个期望为$\theta_i\theta_j\omega_{g_ig_j}$。且为了保持一个组整体上与其他组的连边数的期望不变，$\theta$应该满足：

$$\sum_i\theta_i\delta_{g_i,r}=1 \tag{3}$$

### 基于DC-SBM的社团分类，极大似然法推导
类似于SBM的情况，这里只是多了一个参数$\theta$:

$$\begin{aligned}P(G|\theta, \omega, g)&=\prod_{i<j}\frac{(\theta_i\theta_j\omega_{g_ig_j})^{A_{ij}}}{A_{ij}!}e^{-\theta_i\theta_j\omega_{g_ig_j}}\times \prod_{i}\frac{(\frac{1}{2}\theta_i^2\omega_{g_ig_i})^{A_{ii}/2}}{(A_{ii}/2)!}e^{-\frac{1}{2}\theta_i^2\omega_{g_ig_j}}\\&=\frac{1}{\prod_{i<j}A_{ij}!\prod_i2^{A_{ii}/2}(A_{ii}/2)!}\times\prod_{i<j}(\theta_i\theta_j)^{A_{ij}}\omega_{g_ig_j}^{A_{ij}}e^{-\theta_i\theta_j\omega_{g_ig_j}}\prod_i\theta_i^{A_{ii}}\omega_{g_ig_i}^{A_{ii}/2}e^{-\frac{1}{2}\theta_i^2\omega_{g_ig_i }}\\&=\frac{1}{\prod_{i<j}A_{ij}!\prod_i2^{A_{ii}/2}(A_{ii}/2)!}\times\prod_i(\theta_i)^{\sum_jA_{ij}}\prod_{rs}(\omega_{rs})^{\frac{1}{2}m_{rs}}e^{-\frac{1}{2}\omega_{rs}}\end{aligned}$$

去除多余部分，求对数得：

$$logP(G|\theta, \omega, g)=2\sum_ik_ilog\theta_i+\sum_{rs}(m_{rs}log\omega_{rs}-\omega_{rs})\tag{4}$$

对这个优化的目标函数求导求极值容易的到$\widehat{\omega_{rs}}=m_{rs}$, 而对于$\theta$, 有一个限制为式$(3)$。忽略掉不含$\theta$的后一项，且仅考虑同一组组$r$内的节点，即$\delta_{g_i r}=1$，则可以列出下面的优化问题

$$\begin{aligned}&min -\sum_ik_ilog\theta_i \\ &s.t. \sum_i\theta_i=1\end{aligned}$$

利用拉格朗日乘子法得到：

$$L(\theta, \lambda)=-\sum_ik_ilog\theta_i + \lambda(\sum_i\theta_i-1)\tag{5}$$

对$\theta_i$求导有：

$$\begin{aligned}L_{\theta_i}^`=-\frac{k_i}{\theta_i}+\lambda&=0\\\theta_i&=\frac{k_i}{\lambda}\end{aligned}$$

将其代入$(5)$，得到：

$$L(\theta, \lambda)=-\sum_ik_ilog(\frac{k_i}{\lambda}) + \lambda(\sum_i\frac{k_i}{\lambda}-1)$$

然后对$\lambda$求导有：

$$\begin{aligned}L^`_{\lambda}=\frac{\sum_ik_i}{\lambda}-0-1&=0\\\lambda&=\sum_ik_i\end{aligned}$$

所以$\widehat{\theta_i}=\frac{k_i}{\sum_ik_i}$， 即同一组内，节点度的归一化的值。
将$\widehat{\theta_i}, \widehat{\omega_{rs}}$代入式$(4)$，经过化简，消除常数项{具体见论文}得到：

$$\mathfrak{L}(G|g)=\sum_{rs}m_{rs}log\frac{m_{rs}}{k_rk_s},\ k_r=\sum_{\delta_{g_i, r}}k_i\tag{6}$$

相对于SBM，仅仅是将$n_rn_s$替换为$k_rk_s$，这保留了网络的度分布的信息。



