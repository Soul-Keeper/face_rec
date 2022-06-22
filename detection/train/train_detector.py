import math
import time
import torch
import argparse
import datetime
import numpy as np
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt


from detection.models.configs import cfg_re50, cfg_mnet
from detection.models.RetinaFace import RetinaFace
from detection.utils.prior_box import PriorBox
from detection.utils.multibox_loss import MultiBoxLoss
from detection.utils.widerface import WiderFaceDetection, detection_collate
from detection.utils.augment import preproc

parser = argparse.ArgumentParser()
parser.add_argument('--training_dataset', default='C:\\Users\\nikit\\Desktop\\face_rec\\detection\\train\\data\\widerface\\train\\label.txt', help='Training dataset directory')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='detection/weights', help='Location to save checkpoint models')

args = parser.parse_args()

if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

rgb_mean = (104, 117, 123)  # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder

net = RetinaFace(cfg=cfg)

if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()


def train():
    net.train()
    epoch = 0 + args.resume_epoch

    loss_l_hist = []
    loss_c_hist = []
    loss_landm_hist = []

    print("Loading dataset...")
    dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), save_folder + cfg['name'] + '_epoch_' + str(epoch) + '.pth')
            if epoch != 0:
                loss_l_hist.append(np.float32(loss_l.cpu().detach().numpy()))
                loss_c_hist.append(np.float32(loss_c.cpu().detach().numpy()))
                loss_landm_hist.append(np.float32(loss_landm.cpu().detach().numpy()))
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print(
            'Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
            .format(epoch, max_epoch, (iteration % epoch_size) + 1,
                    epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr,
                    batch_time, str(datetime.timedelta(seconds=eta))))

    torch.save(net.state_dict(), save_folder + cfg['name'].lower() + "_" + cfg['epoch'].lower() + "_" + str(datetime.date.today()) + '.pth')

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.plot(list(range(1, len(loss_l_hist) + 1)), loss_l_hist, label='loss_loc')
    ax1.set_title("loss_loc")
    ax2.plot(list(range(1, len(loss_c_hist) + 1)), loss_c_hist, label='loss_class')
    ax2.set_title("loss_class")
    ax3.plot(list(range(1, len(loss_landm_hist) + 1)), loss_landm_hist, label='loss_landm')
    ax1.set_title("loss_landm")
    plt.savefig(str(cfg['name'].lower() + "_" + str(datetime.date.today()) + "_loss_hist.png"))
    

def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** step_index)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
