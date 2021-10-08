import torch
import torchvision.models as models
import torch.optim as optim
import argparse
import matplotlib.pylab as plt

from network.deeplabv3.deeplabv3 import *
from network.deeplabv2 import *
from build_data import *
from module_list import *


parser = argparse.ArgumentParser(description='Supervised Segmentation with Partial Labels')
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--port', default=None, type=int)

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--lr', default=2.5e-3, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--apply_aug', default='cutout', type=str, help='apply semi-supervised method: cutout cutmix classmix')
parser.add_argument('--weak_threshold', default=0.7, type=float)
parser.add_argument('--strong_threshold', default=0.97, type=float)
parser.add_argument('--output_dim', default=256, type=int, help='output dimension from representation head')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--partial', default='p0', type=str, help='p0, p1, p5, p25')
parser.add_argument('--dataset', default='cityscapes', type=str, help='pascal, cityscapes')
parser.add_argument('--backbone', default='deeplabv3p', type=str, help='choose backbone: deeplabv3p, deeplabv2')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

data_loader = BuildDataLoader(args.dataset, 0)
train_l_loader, test_loader = data_loader.build(supervised=True, partial=args.partial, partial_seed=args.seed)


# Load Semantic Network
device = torch.device("cuda:{:d}".format(args.gpu) if torch.cuda.is_available() else "cpu")
if args.backbone == 'deeplabv3p':
    model = DeepLabv3Plus(models.resnet101(pretrained=True), num_classes=data_loader.num_segments, output_dim=args.output_dim).to(device)
elif args.backbone == 'deeplabv2':
    model = DeepLabv2(models.resnet101(pretrained=True), num_classes=data_loader.num_segments, output_dim=args.output_dim).to(device)

total_epoch = 200
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
scheduler = PolyLR(optimizer, total_epoch, power=0.9)

train_epoch = len(train_l_loader)
test_epoch = len(test_loader)
avg_cost = np.zeros((total_epoch, 6))
iteration = 0
for index in range(total_epoch):
    cost = np.zeros(3)
    train_l_dataset = iter(train_l_loader)

    model.train()
    l_conf_mat = ConfMatrix(data_loader.num_segments)
    for i in range(train_epoch):
        train_l_data, train_l_label = train_l_dataset.next()
        train_l_data, train_l_label = train_l_data.to(device), train_l_label.to(device)

        optimizer.zero_grad()

        # generate labelled and unlabelled data loss
        pred_l, rep_l = model(train_l_data)
        pred_l_large = F.interpolate(pred_l, size=train_l_label.shape[1:], mode='bilinear', align_corners=True)

        # supervised-learning loss
        sup_loss = compute_supervised_loss(pred_l_large, train_l_label)

        loss = sup_loss
        loss.backward()
        optimizer.step()

        l_conf_mat.update(pred_l_large.argmax(1).flatten(), train_l_label.flatten())
        avg_cost[index, 0] += sup_loss.item() / train_epoch
        iteration += 1

    avg_cost[index, 1:3] = l_conf_mat.get_metrics()

    with torch.no_grad():
        model.eval()
        test_dataset = iter(test_loader)
        conf_mat = ConfMatrix(data_loader.num_segments)
        for i in range(test_epoch):
            test_data, test_label = test_dataset.next()
            test_data, test_label = test_data.to(device), test_label.to(device)

            pred, _ = model(test_data)
            pred = F.interpolate(pred, size=test_label.shape[1:], mode='bilinear', align_corners=True)
            loss = compute_supervised_loss(pred, test_label)

            # compute metrics by confusion matrix
            conf_mat.update(pred.argmax(1).flatten(), test_label.flatten())
            avg_cost[index, 3:] += loss.item() / test_epoch

        avg_cost[index, 4:6] = conf_mat.get_metrics()

    scheduler.step()
    print('EPOCH: {:04d} ITER: {:04d} | TRAIN [Loss | mIoU | Acc.]: {:.4f} {:.4f} {:.4f} || Test [Loss | mIoU | Acc.]: {:.4f} {:.4f} {:.4f}'
        .format(index, iteration, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2],
                avg_cost[index][3], avg_cost[index][4], avg_cost[index][5]))
    print('Top: mIoU {:.4f} IoU {:.4f}'.format(avg_cost[:, 4].max(), avg_cost[:, 5].max()))

    if avg_cost[index][4] >= avg_cost[:, 4].max():
        torch.save(model.state_dict(), 'model_weights/{}_{}_sup_{}.pth'.format(args.dataset, args.partial, args.seed))

    np.save('logging/{}_{}_sup_{}.npy'.format(args.dataset, args.partial, args.seed), avg_cost)
