
import os
import itertools
from skimage.io import imsave
import sys
from datetime import datetime
import json

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

from utils import ImagePool, UnalignedImgMaskDataset
from model import InstaGAN
import losses



class Trainer():
    def __init__(self, params):
        self.params = params
        training_params = params["training"]

        self.max_epoch = training_params["max_epoch"]

        self.learning_rate = training_params["learning_rate"]
        self.beta1 = training_params["beta1"]

        self.X_name = training_params["X_name"]
        self.Y_name = training_params["Y_name"]
        self.image_dir = training_params["image_dir"]
        self.annotation_file = training_params["annotation_file"]
        self.ins_max = 8
        self.ins_per = 2
        self.checkpoint_root = training_params["checkpoint_root"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lambda_X = training_params["lambda_X"]
        self.lambda_Y = training_params["lambda_Y"]
        self.lambda_idt = training_params["lambda_idt"]
        self.lambda_ctx = training_params["lambda_ctx"]

        pool_size = 50
        self.fake_X_pool = ImagePool(pool_size)
        self.fake_Y_pool = ImagePool(pool_size)

        dt_now = datetime.now()
        dt_seq = dt_now.strftime("%y%m%d_%H%M")
        self.result_dir = os.path.join("./result", f"{dt_seq}_{self.X_name}2{self.Y_name}")
        self.weight_dir = os.path.join(self.result_dir, "weights")
        self.sample_dir = os.path.join(self.result_dir, "sample")
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.weight_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        
        with open(os.path.join(self.result_dir, "params.json"), mode="w") as f:
            json.dump(params, f)

    def train(self, resume_from=False):

        model = InstaGAN(self.params)
        model.cast_device(self.device)

        # construct dataloader
        train_dataset = UnalignedImgMaskDataset(X_name=self.X_name, Y_name=self.Y_name, simple_resize=False,
                                                image_dir=self.image_dir, annotation_file=self.annotation_file)
        train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=4) 

        # construct optimizers
        optimizer_G = Adam(filter(lambda p: p.requires_grad, itertools.chain(model.G_XY.parameters(), model.G_YX.parameters())),
                           lr=self.learning_rate,
                           betas=(self.beta1, 0.999))
        optimizer_D_X = Adam(model.D_X.parameters(),
                            lr=self.learning_rate*5, 
                            betas=(self.beta1, 0.999))
        optimizer_D_Y = Adam(model.D_Y.parameters(),
                    lr=self.learning_rate*5, 
                    betas=(self.beta1, 0.999))

        start_epoch = 1
        
        # resume
        if resume_from:
            file_list = os.listdir(f"{resume_from}/weights")
            epoch_list = list(set([int(file.split("_")[0]) for file in file_list]))
            latest_epoch = max(epoch_list)
            start_epoch = latest_epoch + 1
            self.max_epoch += latest_epoch
            print(f"Resume training from {resume_from} at epoch {start_epoch}")
            model.G_XY.load_state_dict(torch.load(f"{resume_from}/weights/{latest_epoch}_{self.X_name}2{self.Y_name}.pth"))
            model.G_YX.load_state_dict(torch.load(f"{resume_from}/weights/{latest_epoch}_{self.Y_name}2{self.X_name}.pth"))
            model.D_X.load_state_dict(torch.load(f"{resume_from}/weights/{latest_epoch}_dis_{self.X_name}.pth"))
            model.D_Y.load_state_dict(torch.load(f"{resume_from}/weights/{latest_epoch}_dis_{self.Y_name}.pth"))
            optimizer_G.load_state_dict(torch.load(f"{resume_from}/weights/{latest_epoch}_opt_G.pth"))
            optimizer_D_X.load_state_dict(torch.load(f"{resume_from}/weights/{latest_epoch}_opt_D_{self.X_name}.pth"))
            optimizer_D_Y.load_state_dict(torch.load(f"{resume_from}/weights/{latest_epoch}_opt_D_{self.Y_name}.pth"))


        criterionGAN = losses.LSGAN().to(self.device)
        criterionCyc = torch.nn.L1Loss().to(self.device)
        criterionIdt = torch.nn.L1Loss().to(self.device)
        criterionCtx = losses.WeightedL1Loss().to(self.device)
        ins_iter = self.ins_max // self.ins_per

        # training roop
        for epoch in range(start_epoch, self.max_epoch + 1):
            print(f"epoch {epoch} start")
            for batch in train_dataloader:
                real_image_mask_X, real_image_mask_Y= self.cast_device(batch)
                real_image_X = real_image_mask_X[:, :3, :, :]
                real_image_Y = real_image_mask_Y[:, :3, :, :]
                real_masks_X = real_image_mask_X[:, 3:, :, :]
                real_masks_Y = real_image_mask_Y[:, 3:, :, :]

                fake_mask_X_list = []
                fake_mask_Y_list = []
                
                for i in range(ins_iter):
                    masks_X = real_masks_X[:, self.ins_per*i:self.ins_per*(i+1), :, :]
                    masks_Y = real_masks_Y[:, self.ins_per*i:self.ins_per*(i+1), :, :]

                    remain_instance_X = (masks_X.to("cpu").numpy()).sum() > 0
                    remain_instance_Y = (masks_Y.to("cpu").numpy()).sum() > 0

                    if (not remain_instance_X) and (not remain_instance_Y):
                        continue

                    if remain_instance_X:
                        temp_real_image_mask_X = torch.cat([real_image_X, masks_X], dim=1)
                        fake_image_mask_Y = model.G_XY(temp_real_image_mask_X)
                        fake_image_Y = fake_image_mask_Y[:, :3, :, :]
                        fake_masks_Y = fake_image_mask_Y[:, 3:, :, :]
                        rec_image_mask_X = model.G_YX(fake_image_mask_Y)

                    if remain_instance_Y:
                        temp_real_image_mask_Y = torch.cat([real_image_Y, masks_Y], dim=1)
                        fake_image_mask_X = model.G_YX(temp_real_image_mask_Y)
                        fake_image_X = fake_image_mask_X[:, :3, :, :]
                        fake_masks_X = fake_image_mask_X[:, 3:, :, :]
                        rec_image_mask_Y = model.G_XY(fake_image_mask_X)

                    #
                    # Update Generators
                    #
                    optimizer_G.zero_grad()

                    if remain_instance_X:
                        loss_G_XY = criterionGAN(model.D_Y(torch.cat([fake_image_mask_Y, *fake_mask_Y_list], dim=1)), is_real=True)
                        loss_cyc_XYX = criterionCyc(rec_image_mask_X, temp_real_image_mask_X)
                        loss_idt_X = criterionIdt(model.G_YX(temp_real_image_mask_X), temp_real_image_mask_X) 
                        weight_X = self.get_weight_for_ctx(masks_X, fake_masks_Y) 
                        loss_ctx_XY = criterionCtx(real_image_X, fake_image_Y, weight=weight_X)
                    else:
                        loss_G_XY = 0
                        loss_cyc_XYX = 0
                        loss_idt_X = 0
                        loss_ctx_XY = 0

                    if remain_instance_Y:
                        loss_G_YX = criterionGAN(model.D_X(torch.cat([fake_image_mask_X, *fake_mask_X_list], dim=1)), is_real=True)
                        loss_cyc_YXY = criterionCyc(rec_image_mask_Y, temp_real_image_mask_Y)
                        loss_idt_Y = criterionIdt(model.G_XY(temp_real_image_mask_Y), temp_real_image_mask_Y)
                        weight_Y = self.get_weight_for_ctx(masks_Y, fake_masks_X)
                        loss_ctx_YX = criterionCtx(real_image_Y, fake_image_X, weight=weight_Y)
                    else:
                        loss_G_YX = 0
                        loss_cyc_YXY = 0
                        loss_idt_Y = 0
                        loss_ctx_YX = 0

                    if remain_instance_X or remain_instance_Y:
                        loss_G = loss_G_XY + loss_G_YX + \
                            self.lambda_X * (loss_cyc_XYX + self.lambda_ctx * loss_ctx_XY + self.lambda_idt * loss_idt_Y) + \
                            self.lambda_Y * (loss_cyc_YXY + self.lambda_ctx * loss_ctx_YX + self.lambda_idt * loss_idt_X)

                        loss_G.backward()

                        optimizer_G.step()
                        loss_G = loss_G.item()


                    # Prepare next roop
                    if remain_instance_Y:
                        fake_image_X = fake_image_X.detach()
                        real_image_Y = fake_image_X
                        fake_masks_X = fake_masks_X.detach()
                        fake_mask_X_list.append(fake_masks_X)
                        fake_image_mask_X = torch.cat([fake_image_X, *fake_mask_X_list], dim=1)
                        fake_image_mask_X_D = self.fake_X_pool.query(fake_image_mask_X)

                    if remain_instance_X:
                        fake_image_Y = fake_image_Y.detach()
                        real_image_X = fake_image_Y
                        fake_masks_Y = fake_masks_Y.detach()
                        fake_mask_Y_list.append(fake_masks_Y)
                        fake_image_mask_Y = torch.cat([fake_image_Y, *fake_mask_Y_list], dim=1)
                        fake_image_mask_Y_D = self.fake_Y_pool.query(fake_image_mask_Y)

                    #                
                    # Update Discriminators
                    #
                    if remain_instance_Y:
                        optimizer_D_X.zero_grad()
                        loss_D_X = 0.5 * (criterionGAN(model.D_X(temp_real_image_mask_X), is_real=True) + criterionGAN(model.D_X(fake_image_mask_X_D), is_real=False))
                        loss_D_X.backward()
                        optimizer_D_X.step()
                        loss_D_X = loss_D_X.item()
                    
                    if remain_instance_X:
                        optimizer_D_Y.zero_grad()
                        loss_D_Y = 0.5 * (criterionGAN(model.D_Y(temp_real_image_mask_Y), is_real=True) + criterionGAN(model.D_Y(fake_image_mask_Y_D), is_real=False))
                        loss_D_Y.backward()
                        optimizer_D_Y.step()
                        loss_D_Y = loss_D_Y.item()

                    print(f"loss_D_X: {loss_D_X:.4f}, loss_D_Y: {loss_D_Y:.4f}, loss_G: {loss_G:.4f}")

            # Save weights
            #if epoch % 5 == 0:
            torch.save(model.G_XY.state_dict(), f"{self.weight_dir}/{epoch}_{self.X_name}2{self.Y_name}.pth")
            torch.save(model.G_YX.state_dict(), f"{self.weight_dir}/{epoch}_{self.Y_name}2{self.X_name}.pth")
            torch.save(model.D_X.state_dict(), f"{self.weight_dir}/{epoch}_dis_{self.X_name}.pth")
            torch.save(model.D_Y.state_dict(), f"{self.weight_dir}/{epoch}_dis_{self.Y_name}.pth")
            torch.save(optimizer_G.state_dict(), f"{self.weight_dir}/{epoch}_opt_G.pth")
            torch.save(optimizer_D_X.state_dict(), f"{self.weight_dir}/{epoch}_opt_D_{self.X_name}.pth")
            torch.save(optimizer_D_Y.state_dict(), f"{self.weight_dir}/{epoch}_opt_D_{self.Y_name}.pth")

            train_dataloader.dataset.shuffle()

            #
            # Generate sample images
            #
            if len(train_dataset.test_Ids_X) > 0:
                for idx in range(len(train_dataset.test_Ids_X)):
                    image_masks = train_dataset.get_test_data(idx, "X")
                    image_masks = image_masks.to(self.device)

                    image = image_masks[:, :3, :, :]
                    masks = image_masks[:, 3:, :, :]
                    _, n_masks, _, _ = masks.size()
                    fake_mask_list = []

                    for i in range(n_masks):
                        temp_masks = masks[:, i:i+1, :, :]
                        temp_real_image_mask = torch.cat([image, temp_masks], dim=1)
                        with torch.no_grad():
                            fake_image_mask = model.G_XY(temp_real_image_mask)
                        fake_image = fake_image_mask[:, :3, :, :]
                        fake_masks = fake_image_mask[:, 3:, :, :]
                        image = fake_image
                        fake_mask_list.append(fake_masks)
                    
                    real_image = image_masks[:, :3, :, :]
                    fake_image = fake_image_mask[:, :3, :, :]
                    real_image = real_image.to("cpu").numpy()
                    fake_image = fake_image.to("cpu").numpy()
                    h, w = real_image.shape[2:]
                    result_image = np.zeros((3, 2*h, 2*w))
                    result_image[:, :h, :w] = real_image
                    result_image[:, :h, w:2*w] = fake_image
                    result_image = (result_image + 1) * 127.5
                    real_mask = masks.sum(dim=1).to("cpu").numpy() * 255
                    fake_mask = torch.cat([*fake_mask_list], dim=1)
                    fake_mask = fake_mask.sum(dim=1).to("cpu").numpy() * 255
                    result_image[:, h:2*h, :w] = real_mask
                    result_image[:, h:2*h, w:2*w] = fake_mask
                    result_image = np.transpose(result_image, (1, 2, 0))
                    result_image = result_image.astype(np.uint8)
                    sample_image_file = f"{self.sample_dir}/{epoch}_{self.X_name}2{self.Y_name}_{idx}.png"
                    imsave(sample_image_file, result_image)

            if len(train_dataset.test_Ids_Y) > 0:
                for idx in range(len(train_dataset.test_Ids_Y)):
                    image_masks = train_dataset.get_test_data(idx, "Y")
                    image_masks = image_masks.to(self.device)

                    image = image_masks[:, :3, :, :]
                    masks = image_masks[:, 3:, :, :]
                    _, n_masks, _, _ = masks.size()
                    fake_mask_list = []

                    for i in range(n_masks):
                        temp_masks = masks[:, i:i+1, :, :]
                        temp_real_image_mask = torch.cat([image, temp_masks], dim=1)
                        with torch.no_grad():
                            fake_image_mask = model.G_YX(temp_real_image_mask)
                        fake_image = fake_image_mask[:, :3, :, :]
                        fake_masks = fake_image_mask[:, 3:, :, :]
                        image = fake_image
                        fake_mask_list.append(fake_masks)
                    
                    real_image = image_masks[:, :3, :, :]
                    fake_image = fake_image_mask[:, :3, :, :]
                    real_image = real_image.to("cpu").numpy()
                    fake_image = fake_image.to("cpu").numpy()
                    h, w = real_image.shape[2:]
                    result_image = np.zeros((3, 2*h, 2*w))
                    result_image[:, :h, :w] = real_image
                    result_image[:, :h, w:2*w] = fake_image
                    result_image = (result_image + 1) * 127.5
                    real_mask = masks.sum(dim=1).to("cpu").numpy() * 255
                    fake_mask = torch.cat([*fake_mask_list], dim=1)
                    fake_mask = fake_mask.sum(dim=1).to("cpu").numpy() * 255
                    result_image[:, h:2*h, :w] = real_mask
                    result_image[:, h:2*h, w:2*w] = fake_mask
                    result_image = np.transpose(result_image, (1, 2, 0))
                    result_image = result_image.astype(np.uint8)
                    sample_image_file = f"{self.sample_dir}/{epoch}_{self.Y_name}2{self.X_name}_{idx}.png"
                    imsave(sample_image_file, result_image)


            real_images_X = real_image_mask_X[:, :3, :, :].to("cpu").numpy()
            real_images_Y = real_image_mask_Y[:, :3, :, :].to("cpu").numpy()
            fake_image_X = fake_image_X.to("cpu").numpy()
            fake_image_Y = fake_image_Y.to("cpu").numpy()
            h, w = real_images_X.shape[2:]
            result_image = np.zeros((3, 2*h, 4*w))
            result_image[:, :h, :w] = real_images_X[0, :, :, :]
            result_image[:, :h, w:2*w] = fake_image_Y[0, :, :, :]
            result_image[:, h:2*h, :w] = real_images_Y[0, :, :, :]
            result_image[:, h:2*h, w:2*w] = fake_image_X[0, :, :, :]

            result_image = (result_image + 1) * 127.5
            real_mask_X = real_image_mask_X[:, 3:, :, :].sum(dim=1).to("cpu").numpy() * 255
            real_mask_Y = real_image_mask_Y[:, 3:, :, :].sum(dim=1).to("cpu").numpy() * 255
            fake_mask_X = torch.cat([*fake_mask_X_list], dim=1)
            fake_mask_Y = torch.cat([*fake_mask_Y_list], dim=1)
            fake_mask_X = fake_mask_X.sum(dim=1).to("cpu").numpy() * 255
            fake_mask_Y = fake_mask_Y.sum(dim=1).to("cpu").numpy() * 255
            
            result_image[:, :h, 2*w:3*w] = real_mask_X
            result_image[:, :h, 3*w:4*w] = fake_mask_Y
            result_image[:, h:2*h, 2*w:3*w] = real_mask_Y
            result_image[:, h:2*h, 3*w:4*w] = fake_mask_X

            result_image = np.transpose(result_image, (1, 2, 0))
            result_image = result_image.astype(np.uint8)
            sample_image_file = f"{self.sample_dir}/{epoch}.png"
            imsave(sample_image_file, result_image)

    def get_weight_for_ctx(self, x, y):
        z = torch.cat([x, y], dim=1)
        z = torch.sum(z, dim=1, keepdim=True)
        z = z.clamp(max=1, min=0)
        return 1 - z  # [0,1] -> [1,0]
            
    def cast_device(self, batch):
        image_mask_X = batch[0]
        image_mask_Y = batch[1]
        image_mask_X = image_mask_X.to(self.device)
        image_mask_Y = image_mask_Y.to(self.device)
        return image_mask_X, image_mask_Y


