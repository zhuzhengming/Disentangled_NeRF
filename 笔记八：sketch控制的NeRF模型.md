## 笔记五：sketch控制的NeRF模型

- #### 思路一计划（sketch控制的NeRF模型）：

  - 深入理解NeRF的代码细节
  - 从最基本的codeNeRF上操作latent code
  - 然后参考CLIPNeRF的方法，将sketch的建模加入操作geometry方面的latent code。
  - 尝试对人体的建模
  - 改进方法

- #### 思路二计划（类似headNeRF的方法，加入SMPL）：

  - 参考文献【1】【3】
  - 搭建geometry和texture可控的NeRF
  - 搭建HeadNeRF 并参考SMPL的代码

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