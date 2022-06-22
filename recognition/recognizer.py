import datetime
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import optim
from pathlib import Path
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter
from torchvision import transforms as T

from alignment.alignment import faces_preprocessing
from recognition.models.IR import Backbone, l2_norm
from recognition.models.heads import Arcface
from recognition.models.MobileFaceNet import MobileFaceNet
from recognition.train.train_utils import get_train_loader, separate_bn_paras, get_val_data, get_time, calculate_metrics,\
    gen_plot, hflip_batch


class FaceRecognizer:
    def __init__(self, name='mobilefacenet', weights_path=None, train=False,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 embedding_size=512, net_depth=50, drop_ratio=0.6, threshold=0.5):
        self.name = name
        self.device = device
        self.embedding_size = embedding_size
        self.net_depth = net_depth
        self.drop_ratio = drop_ratio
        self.threshold = threshold
        self.step = 0
        self.batch_size = 128

        if not train:
            print('Loading model for inference...')
            if self.name == 'mobilefacenet':
                self.model = MobileFaceNet(self.embedding_size).to(self.device)
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            elif self.name == 'ir_se':
                self.model = Backbone(net_depth, drop_ratio, 'ir_se').to(self.device)
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            else:
                print("Invalid backbone")
            self.model.eval()
            torch.set_grad_enabled(False)
            print('Model loaded successfully')

        else:
            print('Creating model for training...')
            save_train_path = Path('C:\\Users\\nikit\\Desktop\\face_rec\\recognition\\weights\\' + str(datetime.date.today()))
            if not save_train_path.exists():
                save_train_path.mkdir()

            if self.name == 'mobilefacenet':
                self.model = MobileFaceNet(self.embedding_size).to(self.device)
            elif self.name == 'ir_se':
                self.model = Backbone(net_depth, drop_ratio, 'ir_se').to(self.device)
            else:
                print("Invalid backbone")

            self.milestones = [12, 15, 18]
            self.loader, self.class_num = get_train_loader()
            self.writer = SummaryWriter()

            self.head = Arcface(embedding_size=self.embedding_size, classnum=self.class_num).to(self.device)
            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

            self.lr = 1e-3
            self.momentum = 0.9
            if self.name == 'mobilefacenet':
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                    {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                    {'params': paras_only_bn}
                ], lr=self.lr, momentum=self.momentum)
            else:
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                    {'params': paras_only_bn}
                ], lr=self.lr, momentum=self.momentum)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=40, verbose=True)
            self.ce_loss = CrossEntropyLoss()
            self.board_loss_every = len(self.loader) // 100
            self.evaluate_every = len(self.loader) // 10
            self.save_every = len(self.loader) // 5
            self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(Path("C:\\Users\\nikit\\Desktop\\data\\faces_emore"))
            print('Model created successfully')

    def infer(self, faces, target_embeddings):
        faces = faces_preprocessing(faces)
        embeddings = self.model(faces)
        diff = embeddings.unsqueeze(-1) - target_embeddings.transpose(1, 0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1  # if no match, set idx to -1
        return min_idx, minimum

    def train(self, epochs):
        self.model.train()
        running_loss = 0.
        for e in range(epochs):
            print('epoch {} started'.format(e))
            # if e == self.milestones[0]:
            #     self.schedule_lr()
            # if e == self.milestones[1]:
            #     self.schedule_lr()
            # if e == self.milestones[2]:
            #     self.schedule_lr()
            for imgs, labels in tqdm(iter(self.loader)):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                thetas = self.head(embeddings, labels)
                loss = self.ce_loss(thetas, labels)
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    # self.writer.add_scalar('train_loss', loss_board, self.step)
                    running_loss = 0.

                if self.step % self.evaluate_every == 0 and self.step != 0:
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(self.agedb_30, self.agedb_30_issame)
                    self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(self.lfw, self.lfw_issame)
                    self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(self.cfp_fp, self.cfp_fp_issame)
                    self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)
                    self.model.train()
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(accuracy)

    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)

    def save_state(self, accuracy, extra=None, model_only=False):
        torch.save(self.model.state_dict(),
                   self.save_train_path/('model_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
        if not model_only:
            torch.save(self.head.state_dict(),
                       self.save_train_path/('head_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
            torch.save(self.optimizer.state_dict(),
                       self.save_train_path/('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))

    def evaluate(self, carray, issame, nrof_folds=5, tta=False):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), self.embedding_size])
        with torch.no_grad():
            while idx + self.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + self.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(self.device)) + self.model(fliped.to(self.device))
                    embeddings[idx:idx + self.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + self.batch_size] = self.model(batch.to(self.device)).cpu()
                idx += self.batch_size

            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(self.device)) + self.model(fliped.to(self.device))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.to(self.device)).cpu()
        tpr, fpr, accuracy, best_thresholds = calculate_metrics(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = T.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor