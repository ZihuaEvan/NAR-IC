import os
import sys
import pprint
import random
import time
import tqdm
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

import losses
import models
import datasets
import lib.utils as utils
from optimizer.optimizer import Optimizer
from evaluation.evaler import Evaler
from scorer.scorer import Scorer
from lib.config import cfg, cfg_from_file
import warnings
warnings.filterwarnings("ignore")

class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            np.random.seed(int(cfg.SEED))
            torch.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)
            """
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            """

        self.num_gpus = torch.cuda.device_count()
        self.distributed = self.num_gpus > 1
        if self.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
        self.device = torch.device("cuda")

        self.rl_stage = False
        self.setup_logging()
        self.setup_dataset()
        self.setup_network()
        self.val_evaler = Evaler(
            eval_ids = cfg.DATA_LOADER.VAL_ID,
            att_feats = cfg.DATA_LOADER.VAL_ATT_FEATS,
            eval_annfile = cfg.INFERENCE.VAL_ANNFILE
        )
        self.test_evaler = Evaler(
            eval_ids = cfg.DATA_LOADER.TEST_ID,
            att_feats = cfg.DATA_LOADER.TEST_ATT_FEATS,
            eval_annfile = cfg.INFERENCE.TEST_ANNFILE
        )
        self.scorer = Scorer()

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        if self.distributed and dist.get_rank() > 0:
            return

        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)

        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info('Training with config:')
        self.logger.info(pprint.pformat(cfg))

    def setup_network(self):
        model = models.create(cfg.MODEL.TYPE)

        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                model.to(self.device),
                device_ids = [self.args.local_rank],
                output_device = self.args.local_rank,
                broadcast_buffers = False
            )
        else:
            self.model = torch.nn.DataParallel(model).cuda()

        if self.args.resume > 0:
            self.model.load_state_dict(
                torch.load(self.snapshot_path("caption_model", self.args.resume),
                    map_location=lambda storage, loc: storage)
            )

            print("load over")

        self.load_epoch = -1
        self.load_iteration = -1
        if self.args.load_epoch:
            self.load_epoch = self.args.resume - 1
            self.load_iteration = int(self.args.resume * 113287 / cfg.TRAIN.BATCH_SIZE)
        self.optim = Optimizer(self.model, self.load_iteration)
        if self.args.resume>0:
            self.optim.optimizer.load_state_dict(
                torch.load(self.snapshot_path("optimizer", self.args.resume),
                           map_location=lambda storage, loc: storage)
            )

        self.xe_criterion = losses.create(cfg.LOSSES.XE_TYPE).cuda()

    def setup_dataset(self):
        self.coco_set = datasets.coco_dataset.CocoDataset(
            image_ids_path = cfg.DATA_LOADER.TRAIN_ID,
            input_seq = cfg.DATA_LOADER.INPUT_SEQ_PATH,
            target_seq = cfg.DATA_LOADER.TARGET_SEQ_PATH,
            att_feats_folder = cfg.DATA_LOADER.TRAIN_ATT_FEATS,
            seq_per_img = cfg.DATA_LOADER.SEQ_PER_IMG,
            max_feat_num = cfg.DATA_LOADER.MAX_FEAT
        )

    def setup_loader(self, epoch):
        self.training_loader = datasets.data_loader.load_train(
            self.distributed, epoch, self.coco_set)

    def eval(self, epoch):
        if (epoch + 1) % cfg.SOLVER.TEST_INTERVAL != 0:
            return None
        if self.distributed and dist.get_rank() > 0:
            return None

        val_res = self.val_evaler(self.model, 'val_' + str(epoch + 1))
        self.logger.info('######## Epoch (VAL)' + str(epoch + 1) + ' ########')
        self.logger.info(str(val_res))

        test_res = self.test_evaler(self.model,'test_' + str(epoch + 1))
        self.logger.info('######## Epoch (TEST)' + str(epoch + 1) + ' ########')
        self.logger.info(str(test_res))

        val = 0
        for score_type, weight in zip(cfg.SCORER.TYPES, cfg.SCORER.WEIGHTS):
            val -= val_res[score_type] * weight
        return val

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

    def save_model(self, epoch):
        if (epoch + 1) % cfg.SOLVER.SNAPSHOT_ITERS != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)
        if self.args.resume>0:
            torch.save(self.model.state_dict(), self.snapshot_path("caption_model", self.args.resume+epoch + 1))
            torch.save(self.optim.optimizer.state_dict(), self.snapshot_path('optimizer', self.args.resume+(epoch + 1)))
        else:
            torch.save(self.model.state_dict(), self.snapshot_path("caption_model", epoch+1))
            torch.save(self.optim.optimizer.state_dict(), self.snapshot_path('optimizer',(epoch + 1)))

    def make_kwargs(self, indices, input_seq, target_seq, att_feats):
        seq_mask = (input_seq > 0).type(torch.cuda.LongTensor)
        seq_mask[:,0] += 1
        seq_mask_sum = seq_mask.sum(-1)
        max_len = int(seq_mask_sum.max())
        input_seq = input_seq[:, 0:max_len].contiguous()
        target_seq = target_seq[:, 0:max_len].contiguous()

        kwargs = {
            cfg.PARAM.INDICES: indices,
            cfg.PARAM.INPUT_SENT: input_seq,
            cfg.PARAM.TARGET_SENT: target_seq,
            cfg.PARAM.ATT_FEATS: att_feats
        }
        return kwargs

    def scheduled_sampling(self, epoch):
        if epoch > cfg.TRAIN.SCHEDULED_SAMPLING.START:
            frac = (epoch - cfg.TRAIN.SCHEDULED_SAMPLING.START) // cfg.TRAIN.SCHEDULED_SAMPLING.INC_EVERY
            ss_prob = min(cfg.TRAIN.SCHEDULED_SAMPLING.INC_PROB * frac, cfg.TRAIN.SCHEDULED_SAMPLING.MAX_PROB)
            self.model.module.ss_prob = ss_prob

    def forward_xe(self, att_feats, target_seq,kwargs):
        rl_flag = False
        if rl_flag ==False:
            logit = self.model(att_feats)
            loss = self.xe_criterion(logit, target_seq.long())
        else:
            ids = kwargs[cfg.PARAM.INDICES]
            # 不使用beam search，采样，将输入数据进行扩充
            ids = utils.expand_numpy(ids)
            att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)


            kwargs['BEAM_SIZE'] = 1
            kwargs[cfg.PARAM.ATT_FEATS] = att_feats

            # 采样（采样函数与ruotian luo不一样，来源于XLAN Net）
            seq_sample, logP_sample = self.model.module.decode_rl(**kwargs)
            # 计算Sample生成的句子与GTs之间的CIDEr得分
            rewards_sample, rewards_info_sample = self.scorer(ids, seq_sample.data.cpu().numpy().tolist())

            mask = seq_sample > 0
            mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
            rewards_sample = torch.from_numpy(rewards_sample).cuda().view(-1, 5)
            reward_baseline = (torch.sum(rewards_sample, 1, keepdim=True) - rewards_sample) / (
                        rewards_sample.size()[1] - 1)

            loss = - logP_sample * mask * (rewards_sample - reward_baseline).view(-1, 1)
            loss = torch.sum(loss) / torch.sum(mask)

        return loss

    def train(self):
        self.model.train()

        iteration = self.load_iteration + 1
        for epoch in range(self.load_epoch + 1, cfg.SOLVER.MAX_EPOCH):
            if epoch >= cfg.TRAIN.REINFORCEMENT.START:
                self.rl_stage = True
            self.setup_loader(epoch)
            
            running_loss = .0
            with tqdm.tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(self.training_loader)) as pbar:
                for _, (indices, input_seq, target_seq, att_feats) in enumerate(self.training_loader):
                    input_seq = input_seq.cuda()
                    target_seq = target_seq.cuda()
                    att_feats = att_feats.cuda()
                    kwargs = self.make_kwargs(indices, input_seq, target_seq, att_feats)
                    loss = self.forward_xe(att_feats, target_seq,kwargs)
                    self.optim.zero_grad()
                    loss.backward()
                    utils.clip_gradient(self.optim.optimizer, self.model,
                        cfg.SOLVER.GRAD_CLIP_TYPE, cfg.SOLVER.GRAD_CLIP)
                    self.optim.step()
                    self.optim.scheduler_step('Iter')

                    running_loss += loss.item()
                    pbar.set_postfix(loss='%.2f' % (running_loss / (_ + 1)))
                    pbar.update()
                    iteration += 1
                    if self.distributed:
                        dist.barrier()
            self.save_model(epoch)
            val = self.eval(epoch)
            self.optim.scheduler_step('Epoch', val)
            self.scheduled_sampling(epoch)
            
            if self.distributed:
                dist.barrier()

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', dest='folder', type=str, default='./experiments/NAIC/')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=148)
    parser.add_argument("--load_epoch", action='store_true')


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.folder is not None:
        cfg_from_file(os.path.join(args.folder, 'config.yml'))
    cfg.ROOT_DIR = args.folder

    trainer = Trainer(args)
    trainer.train()