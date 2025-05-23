# STPrompt

这是论文《Weakly Supervised Video Anomaly Detection and Localization with Spatio-Temporal Prompts》的实现，基于VadCLIP代码库。

## 模型概述

STPrompt是一种用于弱监督视频异常检测和定位的模型，它使用时空提示嵌入，基于预训练的视觉语言模型(VLMs)。模型采用双流网络结构，一个关注时间维度，一个关注空间维度，并结合预训练VLMs的知识和原始视频的自然运动先验。

主要特点：
- 基于运动先验的空间注意力聚合(SA2)机制，用于关注潜在的空间异常位置
- 时间CLIP适配器，增强时间上下文捕获能力
- 基于LLM的文本提示的空间异常定位，使用训练无关的方法
- 双分支框架，包括分类分支和对齐分支

## 文件结构

- `stprompt_model.py`: STPrompt模型的实现
- `stprompt_option.py`: 模型配置参数
- `stprompt_train.py`: XD-Violence数据集的训练脚本
- `stprompt_test.py`: XD-Violence数据集的测试脚本
- `stprompt_ucf_train.py`: UCF-Crime数据集的训练脚本
- `stprompt_ucf_test.py`: UCF-Crime数据集的测试脚本

## 使用方法

### 训练

对于XD-Violence数据集：
```
python stprompt_train.py
```

对于UCF-Crime数据集：
```
python stprompt_ucf_train.py
```

### 测试

对于XD-Violence数据集：
```
python stprompt_test.py
```

对于UCF-Crime数据集：
```
python stprompt_ucf_test.py
```

## 主要组件

1. **空间注意力聚合(SA2)机制**：使用运动先验来关注潜在的空间异常位置，通过计算相邻帧之间的差异来识别运动幅度较大的区域。

2. **时间CLIP适配器**：增强时间上下文捕获能力，基于相对距离而非特征相似度的自注意力机制。

3. **基于LLM的文本提示的空间异常定位**：使用训练无关的方法进行空间异常定位，将空间异常定位视为空间块检索过程。

4. **双分支框架**：包括分类分支和对齐分支，分类分支直接预测异常置信度，对齐分支计算异常类别概率。

## 参考

- 原论文：《Weakly Supervised Video Anomaly Detection and Localization with Spatio-Temporal Prompts》
- 基于VadCLIP代码库：https://github.com/NEWYUEQING/3/tree/main/VadCLIP-main
