## 笔记：可解耦控制的NeRF模型

- #### NeRF代码细节：

- 要渲染这个NeRF，有3个步骤:

  - 相机光线通过场景来采样3D点
  - 利用step1中的点及其对应的2D观察方向(θ， φ)作为输入到MLP，得到颜色(c = (r, g, b))和密度σ的输出集
  - 使用体绘制技术将这些颜色和密度累积到一个2D图像中[注:体绘制是指从采样的3D点创建一个2D投影]

- - npz数据包含了RGB图片和c2w的转换矩阵
  - 网络都为全连接层，RELU为激活函数；
  - position encoder
  - 分为fine和coarse

  

- #### 人体建模效果提升参考方法：

  - ##### 【1】Animatable NeRF

  - ##### 【2】Human NeRF

  - ##### 【3】 Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans





# Conditional NeRF 笔记理解

### 训练时：

- 每一个object都训练出一个1*256维度的latent code向量，说明每一个object都有一个latent code。

## 对于已有训练出的模型：

- camera pose的操作是设置azimuth、 elevation 和 distance来得到C2W转换矩阵来进行的。


- latent code 的操作是：优化提取输入图形的latent code  然后保存下来再赋值给模型渲染，便可利用输入图像的latent code
