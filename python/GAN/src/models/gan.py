import os
import sys
import pickle
import random
import argparse
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms # 画像用データセット諸々
from torchvision.utils import save_image

sys.path.append(os.getcwd())

from sub_module import Generator, Discriminator
from logs.set_log import set_log
import utils

logger = set_log('test.log')
SEED = 0
SAVE_EPOCH = [1, 4, 9]
utils.set_seed(SEED)

logger.info('torch.cuda.is_available(): '.format(torch.cuda.is_available()))
logger.debug('SEED: %d' % SEED)


def train_loop(args, 
               discriminator, generator, 
               criterion, optim_dis, optim_gen, 
               data_loader, 
               save_dir):
    history = {'loss_dis':[], 'loss_gen':[]}
    
    for ep in range(args.epoch):
        logger.debug('epoch %d' % (ep+1))
        loss_dis, loss_gen = train(args,
                                   discriminator, generator,
                                   criterion,
                                   optim_dis, optim_gen,
                                   data_loader)
        
        logger.debug('loss_dis %.4f' % loss_dis)
        logger.debug('loss_gen %.4f' % loss_gen)
        
        history['loss_dis'].append(loss_dis)
        history['loss_gen'].append(loss_gen)

        # 特定のエポックでGeneratorから画像を生成してモデルも保存
        if ep in SAVE_EPOCH:
            generate(args, generator, save_dir)
            
            saved_dis = os.path.join(save_dir, 'D_%03d.pth' % (ep+1))
            saved_gen = os.path.join(save_dir, 'G_%03d.pth' % (ep+1))
            torch.save(discriminator.state_dict(), saved_dis)
            torch.save(generator.state_dict(), saved_gen)
        
    # 学習履歴の保存
    with open(os.path.join(save_dir, 'history.pkl'), 'wb') as fo:
        pickle.dump(history, fo)
        logger.info('save history ... {}'.format(os.path.join(save_dir, 'history.pkl')))


def train(args, discriminator, generator, criterion, optim_dis, optim_gen, data_loader):

    discriminator.train()
    generator.train()
    
    batch = args.batch

    y_real = utils.to_cuda(Variable(torch.ones(batch, 1)))
    y_fake = utils.to_cuda(Variable(torch.zeros(batch, 1)))

    loss_dis, loss_gen = 0, 0

    datas = tqdm(data_loader)
    for idx_batch, (real_images, _) in enumerate(datas):
        datas.set_description('Processing DataLoader %d' % idx_batch)

        # 一番最後、バッチサイズに満たない場合は無視する
        if real_images.size()[0] != batch: break

        real_images = utils.to_variable(real_images)
        z = utils.to_variable(torch.rand((batch, args.z_dim)))

        # Discriminatorの更新
        optim_dis.zero_grad()

        # Discriminatorにとって本物画像の認識結果は1（本物）に近いほどよい
        # E[log(D(x))]
        D_real = discriminator(real_images)
        loss_real = criterion(D_real, y_real)

        # DiscriminatorにとってGeneratorが生成した偽物画像の認識結果は0（偽物）に近いほどよい
        # E[log(1 - D(G(z)))]
        # fake_imagesを通じて勾配がGに伝わらないようにdetach()して止める
        fake_images = generator(z)
        D_fake = discriminator(fake_images.detach())
        loss_fake = criterion(D_fake, y_fake)   # size([128,1])

        # 2つのlossの和を最小化する
        loss_dis_batch = loss_real + loss_fake
        loss_dis_batch.backward()
        optim_dis.step()  # これでGのパラメータは更新されない！
        loss_dis += float(loss_dis_batch.data)

        # Generatorの更新
        z = utils.to_variable(torch.rand((batch, args.z_dim)))
        optim_gen.zero_grad()

        # GeneratorにとってGeneratorが生成した画像の認識結果は1（本物）に近いほどよい
        # E[log(D(G(z)))
        fake_images = generator(z)
        D_fake = discriminator(fake_images)
        loss_gen_batch = criterion(D_fake, y_real)
        loss_gen_batch.backward()
        optim_gen.step()
        loss_gen += float(loss_gen_batch.data)
        
        #sys.stdout.write('\033[2K\033[G LOSS -- gen: {:.4f} dis: {:.4f}'.format(float(loss_gen_batch.data), float(loss_dis_batch.data)))
        #sys.stdout.flush()
    
    loss_dis /= len(data_loader)
    loss_gen /= len(data_loader)
    
    return loss_dis, loss_gen


def generate(args, generator, save_dir):
    """
    学習途中のエポックでサンプル画像を生成する
    """
    generator.eval()
    os.makedirs(save_dir, exist_ok=True)

    # 生成のもとになる乱数を生成 # Generatorでサンプル生成
    z_sample = utils.to_variable(torch.rand((64, args.z_dim)), volatile=True)
    samples = generator(z_sample).data.cpu()
    save_png = os.path.join(save_dir, 'epoch_%03d.png' % args.epoch)
    logger.info('generate sample picture ... {}/epoch_{}.png'.format(save_dir, args.epoch))
    save_image(samples, save_png)
    

def plot_loss(save_dir):

    with open(os.path.join(save_dir, 'history.pkl'), 'rb') as fh:
        history = pickle.load(fh)

    sns.set()
    D_loss, G_loss = history['loss_dis'], history['loss_gen']
    plt.plot(D_loss, label='loss_dis')
    plt.plot(G_loss, label='loss_gen')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(save_dir, 'loss.png'))


def create_model_id(args):
    return 'b{}_lr{}_z{}_ep{}'.format(args.batch, args.lr, args.z_dim, args.epoch)

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--z_dim', type=int, default=62)
    parser.add_argument('--epoch', type=int, default=25)
    #parser.add_argument('--sample_num', type=int, default=16)
    parser.set_defaults(no_thres=False)
    return parser


def run():
    parser = create_arg_parser()
    args = parser.parse_args()

    save_dir = create_model_id(args)

    generator = utils.to_cuda(Generator())
    discriminator = utils.to_cuda(Discriminator())

    # optimizer
    optim_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optim_dis = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # loss
    criterion = nn.BCELoss()

    # dataset loader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('datasets/mnist', train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    train_loop(args, discriminator, generator, criterion, optim_dis, optim_gen, data_loader, save_dir)

    plot_loss(save_dir)

if __name__ == '__main__':
    run()
    sys.stdout.write('DONE!\n')
