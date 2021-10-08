import torch
import torchvision.models as models
import torch.optim as optim
import argparse
import matplotlib.pylab as plt

from network.deeplabv3.deeplabv3 import *
from network.deeplabv2 import *
from build_data import *
from module_list import *


parser = argparse.ArgumentParser(description='Semi-supervised Segmentation with Partial Labels')
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--port', default=None, type=int)

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--lr', default=2.5e-3, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--apply_aug', default='cutout', type=str, help='apply semi-supervised method: cutout cutmix classmix')
parser.add_argument('--weak_threshold', default=0.7, type=float)
parser.add_argument('--strong_threshold', default=0.97, type=float)
parser.add_argument('--apply_reco', action='store_true')
parser.add_argument('--num_negatives', default=512, type=int, help='number of negative samples')
parser.add_argument('--num_queries', default=256, type=int, help='number of queries per segment per image')
parser.add_argument('--temp', default=0.5, type=float)
parser.add_argument('--output_dim', default=256, type=int, help='output dimension from representation head')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--partial', default='p0', type=str, help='p0, p1, p5, p25')
parser.add_argument('--dataset', default='pascal', type=str, help='pascal, cityscapes')
parser.add_argument('--backbone', default='deeplabv3p', type=str, help='choose backbone: deeplabv3p, deeplabv2')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

data_loader = BuildDataLoader(args.dataset, 0)  # partial dataset use all labelled images
train_l_loader, train_u_loader, test_loader = data_loader.build(supervised=False, partial=args.partial, partial_seed=args.seed)

# Load Semantic Network
device = torch.device("cuda:{:d}".format(args.gpu) if torch.cuda.is_available() else "cpu")
if args.backbone == 'deeplabv3p':
    model = DeepLabv3Plus(models.resnet101(pretrained=True), num_classes=data_loader.num_segments, output_dim=args.output_dim).to(device)
elif args.backbone == 'deeplabv2':
    model = DeepLabv2(models.resnet101(pretrained=True), num_classes=data_loader.num_segments, output_dim=args.output_dim).to(device)

total_epoch = 200
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
scheduler = PolyLR(optimizer, total_epoch, power=0.9)
ema = EMA(model, 0.99)

train_epoch = len(train_l_loader)
test_epoch = len(test_loader)
avg_cost = np.zeros((total_epoch, 10))

iteration = 0
for index in range(total_epoch):
    cost = np.zeros(3)
    train_l_dataset = iter(train_l_loader)
    train_u_dataset = iter(train_u_loader)

    model.train()
    ema.model.train()
    l_conf_mat = ConfMatrix(data_loader.num_segments)
    u_conf_mat = ConfMatrix(data_loader.num_segments)
    for i in range(train_epoch):
        train_l_data, train_l_label = train_l_dataset.next()
        train_l_data, train_l_label = train_l_data.to(device), train_l_label.to(device)

        train_u_data, train_u_label = train_u_dataset.next()
        train_u_data, train_u_label = train_u_data.to(device), train_u_label.to(device)

        optimizer.zero_grad()

        # generate pseudo-labels
        with torch.no_grad():
            pred_u, _ = ema.model(train_u_data)
            pred_u_large_raw = F.interpolate(pred_u, size=train_u_label.shape[1:], mode='bilinear', align_corners=True)
            pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u_large_raw, dim=1), dim=1)
            pseudo_labels[train_u_label >= 0] = train_u_label[train_u_label >= 0]  # use provided gt label
            pseudo_logits[train_u_label >= 0] = 1.0

            # random scale images first
            train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                batch_transform(train_u_data, pseudo_labels, pseudo_logits,
                                data_loader.crop_size, data_loader.scale_size, apply_augmentation=False)

            # apply mixing strategy: cutout, cutmix or classmix
            train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                generate_unsup_data(train_u_aug_data, train_u_aug_label, train_u_aug_logits, mode=args.apply_aug)

            # apply augmentation: color jitter + flip + gaussian blur
            train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                batch_transform(train_u_aug_data, train_u_aug_label, train_u_aug_logits,
                                data_loader.crop_size, (1.0, 1.0), apply_augmentation=True)

        # generate labelled and unlabelled data loss
        pred_l, rep_l = model(train_l_data)
        pred_l_large = F.interpolate(pred_l, size=train_l_label.shape[1:], mode='bilinear', align_corners=True)

        pred_u, rep_u = model(train_u_aug_data)
        pred_u_large = F.interpolate(pred_u, size=train_l_label.shape[1:], mode='bilinear', align_corners=True)

        rep_all = torch.cat((rep_l, rep_u))
        pred_all = torch.cat((pred_l, pred_u))

        # supervised-learning loss
        sup_loss = compute_supervised_loss(pred_l_large, train_l_label)

        # unsupervised-learning loss
        unsup_loss = compute_unsupervised_loss(pred_u_large, train_u_aug_label, train_u_aug_logits, args.strong_threshold)

        # apply regional contrastive loss
        if args.apply_reco:
            with torch.no_grad():
                train_l_aug_logits, train_l_aug_label = torch.max(torch.softmax(pred_l_large, dim=1), dim=1)
                train_l_aug_label[train_l_label >= 0] = train_l_label[train_l_label >= 0]

                train_u_aug_mask = train_u_aug_logits.ge(args.weak_threshold).float()
                train_l_aug_mask = train_l_aug_logits.ge(0.97).float()
                train_l_aug_mask[train_l_label >= 0] = 1.0

                mask_all = torch.cat((train_l_aug_mask.unsqueeze(1), train_u_aug_mask.unsqueeze(1)))
                mask_all = F.interpolate(mask_all, size=pred_all.shape[2:], mode='nearest')

                label_l = F.interpolate(label_onehot(train_l_aug_label, data_loader.num_segments), size=pred_all.shape[2:], mode='nearest')
                label_u = F.interpolate(label_onehot(train_u_aug_label, data_loader.num_segments), size=pred_all.shape[2:], mode='nearest')
                label_all = torch.cat((label_l, label_u))

                prob_l = torch.softmax(pred_l, dim=1)
                prob_u = torch.softmax(pred_u, dim=1)
                prob_all = torch.cat((prob_l, prob_u))

            reco_loss = compute_reco_loss(rep_all, label_all, mask_all, prob_all, args.strong_threshold,
                                          args.temp, args.num_queries, args.num_negatives)
        else:
            reco_loss = torch.tensor(0.0)

        loss = sup_loss + unsup_loss + reco_loss
        loss.backward()
        optimizer.step()
        ema.update(model)

        l_conf_mat.update(pred_l_large.argmax(1).flatten(), train_l_label.flatten())
        u_conf_mat.update(pred_u_large_raw.argmax(1).flatten(), train_u_label.flatten())

        cost[0] = sup_loss.item()
        cost[1] = unsup_loss.item()
        cost[2] = reco_loss.item()
        avg_cost[index, :3] += cost / train_epoch
        iteration += 1

    avg_cost[index, 3:5] = l_conf_mat.get_metrics()
    avg_cost[index, 5:7] = u_conf_mat.get_metrics()

    with torch.no_grad():
        ema.model.eval()
        test_dataset = iter(test_loader)
        conf_mat = ConfMatrix(data_loader.num_segments)
        for i in range(test_epoch):
            test_data, test_label = test_dataset.next()
            test_data, test_label = test_data.to(device), test_label.to(device)

            pred, _ = ema.model(test_data)
            pred = F.interpolate(pred, size=test_label.shape[1:], mode='bilinear', align_corners=True)
            loss = compute_supervised_loss(pred, test_label)

            conf_mat.update(pred.argmax(1).flatten(), test_label.flatten())

            avg_cost[index, 7] += loss.item() / test_epoch
        avg_cost[index, 8:] = conf_mat.get_metrics()

    scheduler.step()
    print('EPOCH: {:04d} ITER: {:04d} | TRAIN [Loss | mIoU | Acc.]: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} || Test [Loss | mIoU | Acc.]: {:.4f} {:.4f} {:.4f}'
          .format(index, iteration, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2],
                  avg_cost[index][3], avg_cost[index][4], avg_cost[index][5], avg_cost[index][6], avg_cost[index][7], avg_cost[index][8],
                  avg_cost[index][9]))
    print('Top: mIoU {:.4f} IoU {:.4f}'.format(avg_cost[:, 8].max(), avg_cost[:, 9].max()))

    if avg_cost[index][8] >= avg_cost[:, 8].max():
        if args.apply_reco:
            torch.save(ema.model.state_dict(), 'model_weights/{}_{}_semi_{}_reco_{}.pth'.format(args.dataset, args.partial, args.apply_aug, args.seed))
        else:
            torch.save(ema.model.state_dict(), 'model_weights/{}_{}_semi_{}_{}.pth'.format(args.dataset, args.partial, args.apply_aug, args.seed))

    if args.apply_reco:
        np.save('logging/{}_{}_semi_{}_reco_{}.npy'.format(args.dataset, args.partial, args.apply_aug, args.seed), avg_cost)
    else:
        np.save('logging/{}_{}_semi_{}_{}.npy'.format(args.dataset, args.partial, args.apply_aug, args.seed), avg_cost)

