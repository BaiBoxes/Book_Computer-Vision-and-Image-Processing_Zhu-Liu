import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch

# 创建文件夹
# 记录训练过程的图片效果
os.makedirs("Chapter_9/Example_9-9/results/images/", exist_ok=True)
# 训练完成时模型保存的位置
os.makedirs("Chapter_9/Example_9-9/results/save/", exist_ok=True) 
# 下载数据集存放的位置
os.makedirs("Chapter_9/Example_9-9/datasets/mnist", exist_ok=True)

# 超参数配置
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image samples")
opt = parser.parse_args()
print(opt)

# 图像的尺寸和图像的像素面积
img_shape = (opt.channels, opt.img_size, opt.img_size)
img_area = np.prod(img_shape)

# 设置cuda:(cuda:0)
cuda = True if torch.cuda.is_available() else False

# mnist数据集下载
mnist = datasets.MNIST(
    root='Chapter_9/Example_9-9/datasets/', train=True, download=True, transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ), 
)

# 配置数据到加载器
dataloader = DataLoader(mnist, batch_size=opt.batch_size, shuffle=True)

#定义判别器，将图片28x28展开成784，然后通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，最后接sigmoid激活函数得到一个0到1之间的概率进行二分类
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_area, 512),                   ## 输入特征数为784，输出为512
            nn.LeakyReLU(0.2, inplace=True),            ## 进行非线性映射
            nn.Linear(512, 256),                        ## 输入特征数为512，输出为256
            nn.LeakyReLU(0.2, inplace=True),            ## 进行非线性映射
            nn.Linear(256, 1),                          ## 输入特征数为256，输出为1
            nn.Sigmoid(),                               ## sigmoid是一个激活函数，二分类问题中可将实数映射到[0, 1],作为概率值, 多分类用softmax函数
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)            ## 鉴别器输入是一个被view展开的(784)的一维图像:(64, 784)
        validity = self.model(img_flat)                 ## 通过鉴别器网络
        return validity                                 ## 鉴别器返回的是一个[0, 1]间的概率


# 定义生成器，输入一个100维的0～1之间的高斯分布，然后通过第一层线性变换将其映射到256维, 然后通过LeakyReLU激活函数，接着进行一个线性变换，再经过一个LeakyReLU激活函数， 然后经过线性变换将其变成784维，最后经过Tanh激活函数是希望生成的假的图片数据分布, 能够在-1～1之间。
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 模型中间块儿
        def block(in_feat, out_feat, normalize=True):           ## block(in， out )
            layers = [nn.Linear(in_feat, out_feat)]             ## 线性变换将输入映射到out维
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))    ## 正则化
            layers.append(nn.LeakyReLU(0.2, inplace=True))      ## 非线性激活函数
            return layers
        # prod():返回给定轴上的数组元素的乘积:1*28*28=784
        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),       ## 线性变化将输入映射 100 to 128, 正则化, LeakyReLU
            *block(128, 256),                                   ## 线性变化将输入映射 128 to 256, 正则化, LeakyReLU
            *block(256, 512),                                   ## 线性变化将输入映射 256 to 512, 正则化, LeakyReLU
            *block(512, 1024),                                  ## 线性变化将输入映射 512 to 1024, 正则化, LeakyReLU
            nn.Linear(1024, img_area),                          ## 线性变化将输入映射 1024 to 784
            nn.Tanh()                                           ## 将(784)的数据每一个都映射到[-1, 1]之间
        )
    # view():相当于numpy中的reshape，重新定义矩阵的形状:这里是reshape(64, 1, 28, 28)
    def forward(self, z):                                       ## 输入的是(64， 100)的噪声数据
        imgs = self.model(z)                                     ## 噪声数据通过生成器模型
        imgs = imgs.view(imgs.size(0), *img_shape)                 ## reshape成(64, 1, 28, 28)
        return imgs                                              ## 输出为64张大小为(1, 28, 28)的图像


# 创建生成器，判别器对象
generator = Generator()
discriminator = Discriminator()

# 首先需要定义loss的度量方式  （二分类的交叉熵）
criterion = torch.nn.BCELoss()

# 其次定义 优化函数,优化函数的学习率为0.0003
# betas:用于计算梯度以及梯度平方的运行平均值的系数
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 如果有显卡，都在cuda模式中运行
if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion = criterion.cuda()

# 进行多个epoch的训练
for epoch in range(opt.n_epochs):                               ## epoch:50
    for i, (imgs, _) in enumerate(dataloader):                  ## imgs:(64, 1, 28, 28)     _:label(64)
        
        # 训练判别器
        # view(): 相当于numpy中的reshape，重新定义矩阵的形状, 相当于reshape(128，784)  原来是(128, 1, 28, 28)
        imgs = imgs.view(imgs.size(0), -1)                          ## 将图片展开为28*28=784  imgs:(64, 784)
        real_img = Variable(imgs).cuda()                            ## 将tensor变成Variable放入计算图中，tensor变成variable之后才能进行反向传播求梯度
        real_label = Variable(torch.ones(imgs.size(0), 1)).cuda()      ## 定义真实的图片label为1
        fake_label = Variable(torch.zeros(imgs.size(0), 1)).cuda()     ## 定义假的图片的label为0

        # 计算真实图片的损失
        optimizer_D.zero_grad()                                   ## 清零前面的梯度
        validity_real = discriminator(real_img)                   ## 经过判别器得到输出为一个[0, 1]之间的概率
        d_real_loss = criterion(validity_real, real_label)         ## 判别器真实图像的损失
        d_real_loss.backward(retain_graph=True)                    ## 设置retain_graph=True，保持计算图

        # 计算假图片的损失
        z = Variable(torch.randn(imgs.size(0), opt.latent_dim)).cuda()       ## 生成均值为0，标准差为1的128个100维的噪声
        fake_img = generator(z)                                  ## 通过生成器产生的假图片
        validity_fake = discriminator(fake_img)                  ## 判别器输出假图像的概率
        d_fake_loss = criterion(validity_fake, fake_label)       ## 计算假图像的损失
        d_fake_loss.backward(retain_graph=True)                  ## 反向传播求得梯度
        optimizer_D.step()                                       ## 更新参数

        # 训练生成器，只需要保证生成的假图像判别为真
        optimizer_G.zero_grad()                                   ## 清零前面的梯度
        validity = discriminator(fake_img)                        ## 经过判别器得到输出为一个[0, 1]之间的概率
        g_loss = criterion(validity, real_label)                 ## 计算生成器的损失
        g_loss.backward(retain_graph=True)                        ## 反向传播求得梯度
        optimizer_G.step()                                       ## 更新参数

        # 打印损失
        print(f"[Epoch {epoch}/{opt.n_epochs}][Batch {i}/{len(dataloader)}] \
            [D loss: {d_real_loss.item() + d_fake_loss.item():.6f}] \
            [G loss: {g_loss.item():.6f}]")

        # 保存每500个batch生成的图片
        if i % opt.sample_interval == 0:
            save_image(fake_img.data, f"Chapter_9/Example_9-9/results/images/{epoch * len(dataloader) + i}.png", nrow=8, normalize=True)

# 保存模型（生成器与判别器）
torch.save(generator.state_dict(), 'Chapter_9/Example_9-9/results/save/generator.pth')
torch.save(discriminator.state_dict(), 'Chapter_9/Example_9-9/results/save/discriminator.pth') 