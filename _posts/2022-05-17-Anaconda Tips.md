## 常用命令

- 创建环境 `conda create --name envname`

- 查看版本

  `conda --version`

  `conda info`

- 激活环境`conda activate envname`

- 退出环境`conda deactivate`

## 加速安装

先安装mamba：

`conda install conda-forge::mamba`

然后用mamba安装：

`mamba install packagename`

## Jupyter Notebook

- 安装扩展：mamba install -c conda-forge jupyter_contrib_nbextensions

- 让jupyer notebook识别新创建的环境：

  在对应的环境下安装ipykernel：`conda install ipykernel`。

  然后在对应环境下执行命令: `python -m ipykernel install --user --name {env_name}`

  