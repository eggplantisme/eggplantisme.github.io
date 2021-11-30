---
layout: post
title: （环境配置）Windows安装graph-tool
tags: [Environment]
date: 2021-11-30 18:46
---

如何在Windows下配置python环境使得可以安装graph-tool库，太长不看版，装个Linux虚拟机。

下午开始翻出导师上周给的代码，用之前安装好的conda环境的jupyter notebook打开，执行import部分时发现有一些库没有，比如sklearn。安装了之后发现还有一个graph—tool没有。上网搜索此库，在[installation instructions](https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions#debian-ubuntu)中发现在windows上无法直接安装，可以选择使用docker。我开始安装从未使用过的docker，最终失败，白费了一个下午。
### Docker的失败尝试

根据[Win10 家庭中文版安装Docker(Win10家庭版 +Hyper-V+Docker）](https://www.cnblogs.com/temari/p/13188168.html)的经验分享，首先更新了系统，然后启动了Hyper-v，下载安装docker，最后用注册表伪装成专业版骗过docker的检测，成功安装了docker。然后根据上面的[installation instructions](https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions#debian-ubuntu)的docker部分的方法，安装graph-tool，打开jupyter，然而jupyter打开后是空的，即使我是在本地放代码的目录下打开，也依然是空的。我仔细看了下打开jupyter的命令：

```docker run -p 8888:8888 -p 6006:6006 -it -u user -w /home/user tiagopeixoto/graph-tool bash```

前面-p是端口映射，-w是指定的在服务端的工作目录，所以打开后什么都没有。后来看到可以使用-v参数挂载本地目录到服务端，试错了几下后，算是成功用jupyter打开我本地的代码。但运行之后发现，没有sklearn库，也没有pip可以安装。我又尝试[在docker中安装python依赖库/模块](https://cloud.tencent.com/developer/article/1540997)的方法二，成功把安装好sklearn的环境路径挂载在服务端，并设置了环境变量后，发现graph-tool又不好使了，好像是两个库有依赖冲突，在官方的安装指南也有提到在Windows下是会出现这样的情况。gg，忙活一下午，又是更新系统，又是安装docker，又是运行环境，毛用没有。最后转换思路，痛定思痛，写代码还是得有linux环境，有的东西没linux真不行。

### Linux虚拟机的胜利

一开始犹豫是装双系统还是虚拟机，但双系统以前装过，体验很差，linux上各种app都很不方便，要想用word，ppt还得切回windows，虚拟机是个不错的选择。根据[Windows10系统安装Linux虚拟机（Ubuntu）超详细教程](https://blog.csdn.net/weixin_43525386/article/details/108920902)用之前下载好的linux18.04镜像，安装成功。安装过程中吃了个饭。之后解决了几个问题，比如刚安装自带的vim很不好用上下左右删除键都有问题，最好更新成完整版的vim：
``` powershell
$sudo apt-get remove vim-common 
$sudo apt-get install vim
```
还有关于重新安装pip3，更新pip3到最新版本。

接下来就开始按照一开始的官方的安装指南，进行安装，但安装指南这里
``` 
deb [ arch=amd64 ] https://downloads.skewed.de/apt DISTRIBUTION main
DISTRIBUTION can be "bullseye, buster, sid, bionic, eoan, focal, groovy"
```
没有给出明确的选哪个DISTRIBUTION，根据博客[ubuntu18.04下安装graph-tool](https://blog.csdn.net/HUSTHY/article/details/108260470)里，才得知是要选bionic的。接下来顺利安装graph-tool，并用传统的pip3 install方式安装sklearn，且两个库都能import，说明在linux下这俩库没啥冲突。最终成功通过import部分，算是可以开始看代码了。（配置环境真的是十分考验耐心和google技术，配完后觉得不难，但寻找正确的配置方法的过程还是很揪心）