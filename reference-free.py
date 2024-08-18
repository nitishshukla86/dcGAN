
import os
from torch.utils.data import Dataset, DataLoader
import glob
from torchvision.utils import make_grid
from skimage import io
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import warnings
from typing import Optional, Tuple, Union
import torch
from diffusers import UNet2DConditionModel
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
from data_load import FRLL
import torch.nn as nn
from deepface import DeepFace
from models import Generator, Discriminator
import argparse
from accelerate import Accelerator
import numpy as np
import itertools
import time
import datetime
import sys
import random
from torchvision.utils import save_image
from torchvision import datasets
from torch.autograd import Variable
from loss import CrossRoad



parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="amsl_rf", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=24, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=1000, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=20, help="interval between model checkpoints")
parser.add_argument("--eval_interval", type=int, default=5, help="interval between model checkpoints")

opt = parser.parse_args()
print(opt)




transform = transforms.Compose([        
        transforms.ToTensor(),
        transforms.Resize((opt.img_height,opt.img_width),antialias=False),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                                transforms.ToPILImage()
                               ])


trainset = FRLL(train=True, transform=transform)
testset = FRLL(train=False,transform=transform)
dataloader=DataLoader(trainset,batch_size=opt.batch_size,drop_last=True,shuffle=True)
val_dataloader=DataLoader(testset,batch_size=4,drop_last=True,shuffle=True)

from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")





def get_embds(model,img,device,lamb=10.0,replicate=True):
    model = accelerator.unwrap_model(model)
    model=model.to(device)
    img=img.to(device)
    feats=[]
    for image in img:
        image=invTrans(image)
        inputs = processor(images=image, return_tensors="pt").to(device)
        feats.append(model.get_image_features(**inputs))
    feats=torch.stack(feats)
    return lamb*feats.repeat(1, 77, 1) if replicate else lamb*feats

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

criterion_GAN = torch.nn.L1Loss()
criterion_pixelwise = CrossRoad()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Initialize generator and discriminator
generator = Generator(img_size=opt.img_width,in_channels=3,out_channels=6)

discriminator = Discriminator()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))



# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def get_TMR(fmr_values,tmr_values,target_fmr=10):
    threshold_10_fmr = np.percentile(fmr_values, 100-target_fmr)
    true_match_rate = np.sum(tmr_values >= threshold_10_fmr) / len(tmr_values)

    print(f"TMR at {target_fmr}% FMR is: {true_match_rate:.2f}%")
    return true_match_rate

def sample_images(batches_done,model,scores=None):
    """Saves a generated sample from the validation set"""
    batch = next(iter(val_dataloader))
    batch={'B':batch['morphed_image'] ,'A':torch.cat([ batch['img1'],batch['img2']],1)  }
    face_embd=get_embds(model,batch['B'].to('cuda'),'cuda')
    real_A = Variable(batch["B"].type(Tensor))
    real_B = Variable(batch["A"].type(Tensor))
    fake_B = generator(real_A,face_embd)
    fakeb1,fakeb2=torch.split(fake_B.data,3,1)
    realb1,realb2=torch.split(real_B.data,3,1)
    img_sample = torch.cat(( real_A.data,fakeb1,fakeb2, realb1,realb2), -2)
    save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done ), nrow=5, normalize=True)

from AdaFace.face_alignment import align
from AdaFace.inference import load_pretrained_model, to_input
adaface = load_pretrained_model('ir_101').to('cuda')
def ada_embd(path,adaface):
    adaface = accelerator.unwrap_model(adaface)
    aligned_rgb_img = align.get_aligned_face(path)
    bgr_input = to_input(aligned_rgb_img)
    feature, _ = adaface(bgr_input.cuda())
    return feature

# ----------
#  Training=
# ----------




accelerator = Accelerator()
cond_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",cache_dir='./cache').cuda()
generator,discriminator,cond_model,adaface, optimizer_G ,optimizer_D, dataloader,val_dataloader = accelerator.prepare(
     generator,discriminator,cond_model, adaface,optimizer_G ,optimizer_D, dataloader,val_dataloader )
face_models = ["AdaFace"]#,"ArcFace","VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib"]

with torch.inference_mode():
    def eval(adaface,model,loader):
        result=[]
        for comp_model in ["AdaFace"]:
            o1_o2,o1_i1,o2_i2,s1_imposter,s2_imposter,s1_tmr_at_01,s1_tmr_at_1,s1_tmr_at_5,s1_tmr_at_10=[],[],[],[],[],[],[],[],[]
            s2_tmr_at_01,s2_tmr_at_1,s2_tmr_at_5,s2_tmr_at_10=[],[],[],[]
            for i,batch in enumerate(tqdm(loader)):
                choice=random.choice([True,False])
                x='img1' if choice else 'img2'
                y='img2' if choice else 'img1'
                batch_={'B':batch['morphed_image'] ,'A':torch.cat([ batch[x],batch[y]],1)  }


                face_embd=get_embds(cond_model,batch_['B'].to('cuda'),'cuda')
                real_A = Variable(batch_["B"].type(torch.cuda.FloatTensor))
                real_B = Variable(batch_["A"].type(torch.cuda.FloatTensor))

                generated_faces = fake_B = generator(real_A,face_embd)#generator(real_A,timestep=torch.tensor(0).cuda(),encoder_hidden_states=face_embd,return_dict=False)[0].detach()
                gen1,gen2=torch.split(generated_faces.data,3,1)
            

                for m,g1,g2,r1,r2 in zip(batch['morphed_image'],gen1,gen2,batch['img1'],batch['img2']):
                    invTrans(g1).save('g1.jpg')
                    invTrans(g2).save('g2.jpg')
                    invTrans(r1).save('r1.jpg')
                    invTrans(r2).save('r2.jpg')
                    invTrans(m).save('m.jpg')
                    
                    try:
                        
                        g1_embd=ada_embd('g1.jpg',adaface)
                        g2_embd=ada_embd('g2.jpg',adaface)
                        r1_embd=ada_embd('r1.jpg',adaface)
                        r2_embd=ada_embd('r2.jpg',adaface)
                        m_embd=ada_embd('m.jpg',adaface)
                        if (torch.cosine_similarity(g1_embd,r1_embd) +torch.cosine_similarity(g2_embd,r2_embd)).item()<(torch.cosine_similarity(g1_embd,r2_embd) +torch.cosine_similarity(g2_embd,r1_embd)).item():
                            g1_embd,g2_embd=g2_embd,g1_embd
                        o1_o2.append(torch.cosine_similarity(g1_embd,g2_embd)[0].item())
                        o1_i1.append(torch.cosine_similarity(g1_embd,r1_embd)[0].item())
                        o2_i2.append(torch.cosine_similarity(g2_embd,r2_embd)[0].item())
                        s1_imposter.append(torch.cosine_similarity(g1_embd,r2_embd)[0].item())
                        s2_imposter.append(torch.cosine_similarity(g2_embd,r1_embd)[0].item())
                    except:
                        continue


            s1_tmr_at_01,s1_tmr_at_1,s1_tmr_at_5,s1_tmr_at_10=get_TMR(s1_imposter,o1_i1,0.1),get_TMR(s1_imposter,o1_i1,1),get_TMR(s1_imposter,o1_i1,5),get_TMR(s1_imposter,o1_i1,10)
            s2_tmr_at_01,s2_tmr_at_1,s2_tmr_at_5,s2_tmr_at_10=get_TMR(s2_imposter,o2_i2,0.1),get_TMR(s2_imposter,o2_i2,1),get_TMR(s2_imposter,o2_i2,5),get_TMR(s2_imposter,o2_i2,10)
        return s1_tmr_at_10,s2_tmr_at_10


prev_time = time.time()
hits=[]
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        generator.train()
        choice=random.choice([True,False])
        x='img1' if choice else 'img2'
        y='img2' if choice else 'img1'
        batch_={'B':batch['morphed_image'] ,'A':torch.cat([ batch[x],batch[y]],1)  }

        # Model inputs
        real_A = Variable(batch_["B"].type(Tensor))
        real_B = Variable(batch_["A"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        # print(valid.device)
        face_embd=get_embds(cond_model,batch_['B'].to(valid.device),valid.device)
        fake_B = generator(real_A,face_embd)

        pred_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel,hit = criterion_pixelwise(*fake_B.chunk(2,1), *real_B.chunk(2,1))
        hits.append(hit)

        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        # loss_G.backward()
        accelerator.backward(loss_G)

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        # loss_D.backward()
        accelerator.backward(loss_D)
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
                time_left,
            )
        )

        # If at sample interval save image
        # if batches_done % opt.sample_interval == 0:
    if epoch % opt.eval_interval==0 and accelerator.is_main_process:
        s1,s2=eval(adaface,generator,val_dataloader)
        sample_images(epoch,cond_model,[s1,s2])
        generator.train()
        torch.save(generator.state_dict(), "saved_models/%s/generator_%d_%s_%s_%s.pth" % (opt.dataset_name, epoch,str(s1),str(s2),str((s1+s2)/2)))
        torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))
        np.save('hits.npy',np.array(hits))
