# 百度网盘AI大赛：手写文字擦除(赛题二)
* 比赛主页: [手写文字擦除](https://aistudio.baidu.com/aistudio/competition/detail/129/0/introduction)

### 项目简介
本项目为百度网盘AI大赛：手写文字擦除(赛题二)的总结，包含第一名方案的学习和复现。

### 比赛任务
选手需要建立模型，对比赛给定的带有手写痕迹的试卷图片进行处理，擦除相关的笔，还原图片原本的样子，并提交模型输出的结果图片。希望各位参赛选手结合当下前沿的计算机视觉技术与图像处理技术，在设计搭建模型的基础上，提升模型的训练性能、精度效果和泛化能力。在保证效果精准的同时，可以进一步考虑模型在实际应用中的性能表现，如更轻量、更高效等。

### 数据集介绍
在本次比赛最新发布的数据集中，所有的图像数据均由真实场景采集得到，再通过技术手段进行相应处理，生成可用的脱敏数据集。该任务为image-to-image的形式，因此源数据和GT数据均以图片的形式来提供。各位选手可基于本次比赛最新发布的训练数据快速融入比赛，为达到更好的算法效果，本次比赛不限制大家使用额外的训练数据来优化模型。测试数据集的GT不做公开，请各位选手基于本次比赛最新发布的测试数据集提交对应的结果文件。

### 数据集构成
```
|- root  
    |- images
    |- gts
```
本次比赛最新发布的数据集共包含训练集、A榜测试集、B榜测试集三个部分，其中训练集共1000个样本，A榜测试集共200个样本，B榜测试集共200个样本；
images 为带摩尔纹的源图像数据，gts 为无摩尔纹的真值数据（仅有训练集数据提供gts ，A榜测试集、B榜测试集数据均不提供gts）；
images 与 gts 中的图片根据图片名称一一对应。
* 训练集: [下载](https://staticsns.cdn.bcebos.com/amis/2021-12/1639027952730/dehw_train_dataset.zip)
* A榜测试集: [下载](https://staticsns.cdn.bcebos.com/amis/2021-12/1639027468553/dehw_testA_dataset.zip)
* B榜测试集: [下载](https://staticsns.cdn.bcebos.com/amis/2022-1/1642677967477/dehw_testB_dataset.zip)

## 数据预处理
利用gt图生成训练图中的笔迹,并将笔迹转化为二值化mask供训练。
```
    src_image = cv2.imread(input_img[i])
    gt_image = cv2.imread(gt_img[i])
    diff_image = np.abs(src_image.astype(np.float32) - gt_image.astype(np.float32))
    mean_image = np.mean(diff_image, axis=-1)
    mask = np.greater(mean_image, threshold).astype(np.uint8)
    mask = mask*255
    mask = np.array([mask for i in range(3)]).transpose(1,2,0)
```

## 模型选择——PERT
场景文本删除(STR)包含两个过程：文本定位和背景重建。现有方法存在两个问题：
* 隐式擦除指导导致对非文本区域的过度擦除；
* 单阶段擦除缺乏对文本区域的彻底去除。

PERT模型的特点：1) 显式的文本擦除；2)平衡的多阶段文本擦除

### 模型结构
![image](https://user-images.githubusercontent.com/62683546/156002760-00e4dc5c-36b7-40a4-aacc-eef501a4b321.png)

整体结构包含**Backbone(共享特征提取)、TLN(Mask生成分支)、BRN(填充图生成分支)、RegionMS(显式擦除)** 四个模块。

#### TLN
TLN模块中，为了生成强鲁棒的文本mask，有效融合多尺度的特征，引入了**PSPMoudle**。

![PSP](https://user-images.githubusercontent.com/62683546/156004684-3c03128c-7d53-4f40-9056-4d0f56100d25.png)

#### BRN
BRN模块为了学习背景纹理的鲁棒重构规则，BRN模块引入了跳跃连接，来学习以下两种特征：
* 对背景和前景纹理感知的低级纹理信息进行建模。
* 捕获高级语义以增强特征表示增强。

此外，为提升性能，在训练过程中，还构建了**多尺度重建模块(MRM)** 来预测多尺度擦除结果(P1输出和P2输出)。

![image](https://user-images.githubusercontent.com/62683546/156004768-1942d4f4-9d6d-4a54-bb21-707c8d4ab9cb.png)

#### RegionMS

![image](https://user-images.githubusercontent.com/62683546/156005057-e2a2b790-2dc5-4681-8fae-b345fdf34ffe.png)

#### 训练机制
**迭代学习**:

![image](https://user-images.githubusercontent.com/62683546/156004911-a578bdc3-bbde-4544-8575-b721e186c50a.png)

**损失函数**:
在Mask生成分支中，由于正反例像素点数量极不平衡，采用**Dice Loss**进行训练：

![image](https://user-images.githubusercontent.com/62683546/156005372-18bfa675-b7cc-483d-b977-80a82a0f99cd.png)




