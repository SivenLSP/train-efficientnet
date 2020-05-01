import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from efficientnet_pytorch import EfficientNet

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
args = parser.parse_args()

EPOCH = 135
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128
LR = 0.001

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，再随机裁剪成32*32 切割中心点的位置随机选取。size可以是tuple也可以是int
    transforms.RandomHorizontalFlip(),  # 随机水平翻转给定的PIL.Image,概率为0.5。即：一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train) #训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取
# print(len(trainloader))  # 391
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# print(len(testloader))  # 100

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型微调 调整输出filters
net = EfficientNet.from_pretrained("efficientnet-b0").to(device)
feature = net._fc.in_features
net._fc = nn.Linear(in_features=feature,out_features=10,bias=True)
# feature = net._fc.out_features
# output_filters = net._conv_stem.out_channels
# net._conv_stem = nn.Conv2d(1, output_filters, kernel_size=3, stride=2, bias=False)
# print(net)

loss_function = nn.CrossEntropyLoss()  # 交叉熵用于多分类
optimizer = optim.Adam(net.parameters(), lr=LR)

# 训练
if __name__ == "__main__":
    best_acc = 85  # 初始化best test accuracy
    print("Start Training")
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                start1 = time.time()
                print('\nEpoch: {}'.format(epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    start2 = time.time()
                    length = len(trainloader)  # 391
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()  # 更新所有的参数，梯度被计算好后，就可以调用这个函数。

                    sum_loss += loss.item()  # item() 用于将一个零维张量转换成浮点数
                    _, predicted = torch.max(outputs.data, 1)  # 取得分最高的那个类 (outputs.data的索引号)
                    total += labels.size(0)  #
                    correct += predicted.eq(labels.data).cpu().sum()  #
                    print('[epoch:{}, iter:{}] Loss: {} | Acc: {}% | Time: {}'.format(epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total, time.time()-start2))
                    f2.write('[epoch:{}, iter:{}] Loss: {} | Acc: {}% '.format(epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()
                print("epoch training time:{} s".format(time.time() - start1))
                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():  # 将不想被追踪梯度的代码块包裹起来，评估模型常用，评估时，并不需要计算可训练参数（requires_grad=True）的梯度。
                    correct = 0  # 清零
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('测试分类准确率为：{}%'.format(100 * correct / total))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net.state_dict(), '{}/net_{}.pth'.format(args.outf, epoch + 1))
                    f.write("EPOCH={},Accuracy= {}%".format(epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH={},best_acc= {}%".format(epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, Total Epoch={}".format(EPOCH))
