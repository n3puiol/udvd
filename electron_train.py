import argparse
import logging
import sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

import data, models, utils


def main(args):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    utils.setup_experiment(args)
    utils.init_logging(args)

    # Build data loaders, a model and an optimizer
    model = models.build_model(args).to(device)
    cpf = model.c # channels per frame
    mid = args.n_frames // 2
    model = nn.DataParallel(model)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.5)
    logging.info(f"Built a model consisting of {sum(p.numel() for p in model.parameters()):,} parameters")

    if args.resume_training:
        state_dict = utils.load_checkpoint(args, model, optimizer, scheduler)
        global_step = state_dict['last_step']
        start_epoch = int(state_dict['last_step']/(403200/state_dict['args'].batch_size))+1
    else:
        global_step = 0
        start_epoch = 0

    train_loader, valid_loader = data.build_dataset(args.dataset, args.data_path, batch_size=args.batch_size, image_size=args.image_size, stride=args.stride, n_frames=args.n_frames, aug=args.aug)

    # Track moving average of loss values
    train_meters = {name: utils.RunningAverageMeter(0.98) for name in (["train_loss", "train_psnr", "train_ssim"])}
    valid_meters = {name: utils.AverageMeter() for name in (["valid_loss"])}
    writer = SummaryWriter(log_dir=args.experiment_dir) if not args.no_visual else None

    for epoch in range(start_epoch, args.num_epochs):
        if args.resume_training:
            if epoch %10 == 0:
                optimizer.param_groups[0]["lr"] /= 2
                print('learning rate reduced by factor of 2')

        train_bar = utils.ProgressBar(train_loader, epoch)
        for meter in train_meters.values():
            meter.reset()

        for batch_id, noisy_inputs in enumerate(train_bar):
#             if batch_id > 10:
#                 break
            model.train()

            global_step += 1
            
            noisy_inputs = noisy_inputs.to(device)

            outputs, est_sigma = model(noisy_inputs)

            noisy_frame = noisy_inputs[:, (mid*cpf):((mid+1)*cpf), :, :]

            if args.blind_noise:
                loss = utils.loss_function(outputs, noisy_frame, mode=args.loss, sigma=est_sigma, device=device)
            else:
                loss = utils.loss_function(outputs, noisy_frame, mode=args.loss, sigma=args.noise_std/255, device=device)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            if args.loss == "loglike":
                with torch.no_grad():
                    if args.blind_noise:
                        outputs, mean_image = utils.post_process(outputs, noisy_frame, model=args.model, sigma=est_sigma, device=device)
                    else:
                        outputs, mean_image = utils.post_process(outputs, noisy_frame, model=args.model, sigma=args.noise_std/255, device=device)

            train_psnr = utils.psnr(noisy_inputs[:, (mid*cpf):((mid+1)*cpf), :, :], outputs)
            train_ssim = utils.ssim(noisy_inputs[:, (mid*cpf):((mid+1)*cpf), :, :], outputs)
            train_meters["train_loss"].update(loss.item())
            train_meters["train_psnr"].update(train_psnr.item())
            train_meters["train_ssim"].update(train_ssim.item())

            if args.loss == "loglike":
                train_bar.log(dict(**train_meters, **mean_meters, lr=optimizer.param_groups[0]["lr"]), verbose=True)
            else:
                train_bar.log(dict(**train_meters, lr=optimizer.param_groups[0]["lr"]), verbose=True)

            if writer is not None and global_step % args.log_interval == 0:
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("loss/train", loss.item(), global_step)
                writer.add_scalar("psnr/train", train_psnr.item(), global_step)
                writer.add_scalar("ssim/train", train_ssim.item(), global_step)
                gradients = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None], dim=0)
                writer.add_histogram("gradients", gradients, global_step)
                sys.stdout.flush()

            if (batch_id+1) % 200 == 0:
                if args.loss == "loglike":
                    logging.info(train_bar.print(dict(**train_meters, **mean_meters, lr=optimizer.param_groups[0]["lr"]))+f" | {batch_id+1} mini-batches ended")
                else:
                    logging.info(train_bar.print(dict(**train_meters, lr=optimizer.param_groups[0]["lr"]))+f" | {batch_id+1} mini-batches ended")

            if (batch_id+1) % 2000 == 0:
                model.eval()
                for meter in valid_meters.values():
                    meter.reset()
                    
                valid_bar = utils.ProgressBar(valid_loader)
                for sample_id, noisy_inputs in enumerate(valid_bar):
                    with torch.no_grad():
                        noisy_inputs = noisy_inputs.to(device)
                        
                        outputs, est_sigma = model(noisy_inputs)
                        
                        noisy_frame = noisy_inputs[:, (mid*cpf):((mid+1)*cpf), :, :]
                        
                        if args.blind_noise:
                            loss = utils.loss_function(outputs, noisy_frame, mode=args.loss, sigma=est_sigma, device=device)
                        else:
                            loss = utils.loss_function(outputs, noisy_frame, mode=args.loss, sigma=args.noise_std/255, device=device)
                        
                        if args.loss == "loglike":
                            if args.blind_noise:
                                outputs, mean_image = utils.post_process(outputs, noisy_frame, model=args.model, sigma=est_sigma, device=device)
                            else:
                                outputs, mean_image = utils.post_process(outputs, noisy_frame, model=args.model, sigma=args.noise_std/255, device=device)
                         
                        valid_meters["valid_loss"].update(loss.item())
                        
                if writer is not None:
                    writer.add_scalar("loss/valid", valid_meters['valid_loss'].avg, global_step)
                    sys.stdout.flush()
                
                logging.info("EVAL:"+train_bar.print(dict(**valid_meters, lr=optimizer.param_groups[0]["lr"])))
                
                utils.save_checkpoint(args, global_step, model, optimizer, score=valid_meters["valid_loss"].avg, mode="min")

        scheduler.step()

        if args.loss == "loglike":
            logging.info(train_bar.print(dict(**train_meters, **mean_meters, lr=optimizer.param_groups[0]["lr"])))
        else:
            logging.info(train_bar.print(dict(**train_meters, lr=optimizer.param_groups[0]["lr"])))

        if (epoch+1) % args.valid_interval == 0:
            model.eval()
            for meter in valid_meters.values():
                meter.reset()

            valid_bar = utils.ProgressBar(valid_loader)
            for sample_id, noisy_inputs in enumerate(valid_bar):
                with torch.no_grad():
                    noisy_inputs = noisy_inputs.to(device)

                    outputs, est_sigma = model(noisy_inputs)

                    noisy_frame = noisy_inputs[:, (mid*cpf):((mid+1)*cpf), :, :]

                    if args.blind_noise:
                        loss = utils.loss_function(outputs, noisy_frame, mode=args.loss, sigma=est_sigma, device=device)
                    else:
                        loss = utils.loss_function(outputs, noisy_frame, mode=args.loss, sigma=args.noise_std/255, device=device)

                    if args.loss == "loglike":
                        if args.blind_noise:
                            outputs, mean_image = utils.post_process(outputs, noisy_frame, model=args.model, sigma=est_sigma, device=device)
                        else:
                            outputs, mean_image = utils.post_process(outputs, noisy_frame, model=args.model, sigma=args.noise_std/255, device=device)

                    valid_meters["valid_loss"].update(loss.item())

            if writer is not None:
                writer.add_scalar("loss/valid", valid_meters['valid_loss'].avg, global_step)
                sys.stdout.flush()

            logging.info("EVAL:"+train_bar.print(dict(**valid_meters, lr=optimizer.param_groups[0]["lr"])))
                
            utils.save_checkpoint(args, global_step, model, optimizer, score=valid_meters["valid_loss"].avg, mode="min")

    logging.info(f"Done training! Best PSNR {utils.save_checkpoint.best_score:.3f} obtained after step {utils.save_checkpoint.best_step}.")


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument("--data-path", default="data", help="path to data directory")
    parser.add_argument("--dataset", default="Nanoparticles", help="train dataset name")
    parser.add_argument("--aug", default=0, type=int, help="augmentations")
    parser.add_argument("--batch-size", default=8, type=int, help="train batch size")
    parser.add_argument("--image-size", default=128, type=int, help="image size for train")
    parser.add_argument("--n-frames", default=5, type=int, help="number of frames for training")
    parser.add_argument("--stride", default=64, type=int, help="stride for patch extraction")

    # Add model arguments
    parser.add_argument("--model", default="blind-video-net-4", help="model architecture")

    # Add loss function
    parser.add_argument("--loss", default="mse", help="loss function used for training")

    # Add noise arguments
    parser.add_argument("--noise_dist", default="G", help="G - Gaussian, P - Poisson")
    parser.add_argument("--noise_mode", default="S", help="B - Blind, S - one noise level")
    parser.add_argument('--noise_std', default = 30, type = float,
                        help = 'noise level when mode is S')
    parser.add_argument('--min_noise', default = 0, type = float,
                        help = 'minimum noise level when mode is B')
    parser.add_argument('--max_noise', default = 100, type = float,
                        help = 'maximum noise level when mode is B')

    # Add optimization arguments
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--num-epochs", default=100, type=int, help="force stop training at specified epoch")
    parser.add_argument("--valid-interval", default=1, type=int, help="evaluate every N epochs")
    parser.add_argument("--save-interval", default=1, type=int, help="save a checkpoint every N steps")

    # Parse twice as model arguments are not known the first time
    parser = utils.add_logging_arguments(parser)
    args, _ = parser.parse_known_args()
    models.MODEL_REGISTRY[args.model].add_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
