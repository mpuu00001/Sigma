import torch      
from torch.utils.data import DataLoader
from models.sigma import Sigma 
import utils as utils
# import tools as utils_deepspeed
from datasets import S2T_Dataset
import os
import time
import argparse, json, datetime
from pathlib import Path
import math
from timm.optim import create_optimizer
from models.models import get_requires_grad_dict
from transformers import get_scheduler
from config import *

def main(args):
    print(args.output_dir)
    utils.init_distributed_mode(args) if args.distributed else None
    print("Distributed mode:", args.distributed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("The device is:", device)
    args.device = device

    print(args)
    utils.set_seed(args.seed)

    print(f"Creating dataset:")        
    train_data = S2T_Dataset(path=train_label_paths[args.dataset], 
                                args=args, phase='train')
    print(train_data)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,shuffle=True)
    else:
        train_sampler = torch.utils.data.SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers, 
                                    collate_fn=train_data.collate_fn,
                                    sampler=train_sampler, 
                                    pin_memory=args.pin_mem,
                                    drop_last=True)

    if args.dataset not in ['How2Sign', 'NationalCSL-DP']:
        this_phase = 'val' if 'MSASL' in args.dataset else 'dev'
        dev_data = S2T_Dataset(path=dev_label_paths[args.dataset], 
                                args=args, phase=this_phase)
        print(dev_data)
        dev_sampler = torch.utils.data.SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers, 
                                    collate_fn=dev_data.collate_fn,
                                    # suffle=True
                                    sampler=dev_sampler, 
                                    pin_memory=args.pin_mem)
        
    test_data = S2T_Dataset(path=test_label_paths[args.dataset], 
                            args=args, phase='test')
    print(test_data)        
    test_sampler = torch.utils.data.SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers, 
                                    collate_fn=test_data.collate_fn,
                                    sampler=test_sampler, 
                                    pin_memory=args.pin_mem)

    print(f"Creating model:")
    vlp_model = Sigma(args)
    vlp_model.to(device)
    vlp_model.train()
    for _, param in vlp_model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    vlp_model_wo_ddp = vlp_model
    if args.distributed:
        vlp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(vlp_model)
        vlp_model = torch.nn.parallel.DistributedDataParallel(vlp_model, device_ids=[args.gpu], find_unused_parameters=True)
        vlp_model_wo_ddp = vlp_model.module
    n_parameters = utils.count_parameters_in_MB(vlp_model_wo_ddp)
    print(f'number of params: {n_parameters}M')

    # Define different learning rate for sgt_decoder and main 
    def param_group_fn(model):
        main_param_group = []  
        sgt_dec_param_group = []  

        for name, param in model.named_parameters():
            if "sgt_dec" in name:  
                sgt_dec_param_group.append(param)  
            elif "sgt_dec_lm_head" in name:
                sgt_dec_param_group.append(param)
            else:
                main_param_group.append(param) 

        return [
            {'params': main_param_group, 'lr': args.lr},  
            {'params': sgt_dec_param_group, 'lr': args.sgt_dec_lr}  
        ]
    optimizer = create_optimizer(args, param_group_fn(vlp_model_wo_ddp))
    lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=optimizer,
                num_warmup_steps=int(args.warmup_epochs * len(train_dataloader)/args.gradient_accumulation_steps),
                num_training_steps=int(args.epochs * len(train_dataloader)/args.gradient_accumulation_steps),
            )

    if args.use_deepspeed:
        vlp_model, optimizer, lr_scheduler = utils.init_deepspeed(args, vlp_model, optimizer, lr_scheduler)
        vlp_model_wo_ddp = vlp_model.module.module
    print(optimizer)

    if args.eval:
        if utils.is_main_process():
            if args.task != "ISLR":
                print("DEV result")
                evaluate_vlp(args, dev_dataloader, vlp_model, vlp_model_wo_ddp, phase='dev')
            if args.get_test_results or args.task == "ISLR":
                print("TEST result")
                evaluate_vlp(args, test_dataloader, vlp_model, vlp_model_wo_ddp, phase='test')
        exit(0)

    if args.batch_size <=1: 
        raise Exception("Batch size should be greater than 1")

    print(f"Start training for {args.epochs} epochs")
    vlp_model = vlp_model.to(torch.float32)  # Convert to float32
    output_dir = Path(args.output_dir)

    start_time = time.time()
    for epoch in range(0, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch_vlp(args, vlp_model, train_dataloader, optimizer, epoch)
    
        if args.save_some_checkpoint and (epoch+1) in args.save_epochs_lst:
            if args.output_dir:
                checkpoint_paths = [output_dir / f'checkpoint_{epoch}.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': get_requires_grad_dict(vlp_model_wo_ddp),
                    }, checkpoint_path)
        elif (epoch+1) == args.epochs:
            if args.output_dir:
                checkpoint_paths = [output_dir / f'checkpoint_{epoch}.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': get_requires_grad_dict(vlp_model_wo_ddp),
                    }, checkpoint_path)
        elif args.save_all_checkpoints:
            if args.output_dir:
                checkpoint_paths = [output_dir / f'checkpoint_{epoch}.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': get_requires_grad_dict(vlp_model_wo_ddp),
                    }, checkpoint_path)

        dev_stats, test_stats = None, None
        # single gpu inference
        if utils.is_main_process():
            if (args.dataset not in ['How2Sign', 'NationalCSL-DP']):
                dev_stats = evaluate_vlp(args, dev_dataloader, vlp_model, vlp_model_wo_ddp, phase='dev')
            test_stats = evaluate_vlp(args, test_dataloader, vlp_model, vlp_model_wo_ddp, phase='test')
            
            if dev_stats is not None:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'dev_{k}': v for k, v in dev_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        if args.debug:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('Done!')

def toggle_params(step, vlp_model):
    if step % 2 != 0:
        for param in vlp_model.sgt_dec.parameters():
            param.requires_grad = False
        for param in vlp_model.sgt_dec_lm_head.parameters():
            param.requires_grad = False
    else: 
        for param in vlp_model.sgt_dec.parameters():
            param.requires_grad = True
        for param in vlp_model.sgt_dec_lm_head.parameters():
            param.requires_grad = True

def train_one_epoch_vlp(args, vlp_model, data_loader, optimizer, epoch):
    vlp_model.train()

    metric_logger = utils.MetricLogger(delimiter="  ", omit='lr')
    metric_logger.add_meter('main_lr', utils.SmoothedValue(window_size=1, fmt='{value:.5f}'))
    metric_logger.add_meter('sgt_dec_lr', utils.SmoothedValue(window_size=1, fmt='{value:.7f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10
    optimizer.zero_grad()
    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for key in src_input.keys():
            if isinstance(src_input[key], torch.Tensor):
                src_input[key] = src_input[key].to(torch.float32).cuda()

        if args.task == "CSLR":
            tgt_input['gt_sentence'] = tgt_input['gt_gloss']
        
        toggle_params(step, vlp_model)
        stack_out = vlp_model(src_input, tgt_input)        
        stc_ll_loss, stc_gl_loss, stm_loss, lm_loss = stack_out['loss_local_stc'], stack_out['loss_global_stc'], stack_out['loss_stm'], stack_out['loss_lm']
        hal_loss = ((1-args.alpha) * stc_gl_loss + args.alpha * (stc_ll_loss)) 
        sgt_loss = ((1-args.beta) * stm_loss + args.beta * (lm_loss))  

        if args.ablate == 'HAL':
            total_loss = sgt_loss
        elif args.ablate == 'SGT':
            total_loss = hal_loss
        else:
            total_loss = (hal_loss + sgt_loss)/2

        if args.use_deepspeed:
            vlp_model.backward(total_loss)
        else:
            total_loss.backward()   

        torch.nn.utils.clip_grad_norm_(vlp_model.parameters(), max_norm=5, norm_type=float('inf'), error_if_nonfinite=False)
        if not math.isfinite(total_loss.item()):
            print(f"Warning: Loss contains {total_loss.item()}! Skipping this batch.")
            optimizer.zero_grad()  
            continue  

        if args.use_deepspeed:
            vlp_model.step()
        else:
            optimizer.step()

        #     masked_tgt = utils.noise_injecting(tgt_input['gt_sentence'], args.noise_rate, args.noise_type, args.random_shuffle, is_train=True)
        #     maskted_tgt_input = vlp_model_wo_ddp.mt5_tokenizer(masked_tgt, return_tensors="pt", padding=True, truncation=True, max_length=50).to(args.device) 

        metric_logger.update(total_loss=total_loss.item())
        metric_logger.update(hal_loss=hal_loss)
        metric_logger.update(sgt_loss=sgt_loss)
        metric_logger.update(main_lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(sgt_dec_lr=optimizer.param_groups[1]["lr"])
        
        if args.debug:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate_vlp(args, data_loader, vlp_model, vlp_model_wo_ddp, phase):
    vlp_model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ", omit='lr')
    header = phase.upper() + ':'

    with torch.no_grad():
 
        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            for key in src_input.keys():
                if isinstance(src_input[key], torch.Tensor):
                    src_input[key] = src_input[key].to(torch.float32).cuda()

            if args.task == "CSLR":
                tgt_input['gt_sentence'] = tgt_input['gt_gloss']
            stack_out = vlp_model(src_input, tgt_input)        
            stc_ll_loss, stc_gl_loss, stm_loss, lm_loss = stack_out['loss_local_stc'], stack_out['loss_global_stc'], stack_out['loss_stm'], stack_out['loss_lm']
            hal_loss = ((1-args.alpha) * stc_gl_loss + args.alpha * (stc_ll_loss)) 
            sgt_loss = ((1-args.beta) * stm_loss + args.beta * (lm_loss))  
            total_loss = (hal_loss + sgt_loss)/2

            metric_logger.update(total_loss=total_loss.item())
            metric_logger.update(hal_loss=hal_loss)
            metric_logger.update(sgt_loss=sgt_loss)
            if args.debug:
                break
    
    metric_logger.synchronize_between_processes()
    print("* Averaged stats:", metric_logger)
    print('* ' + phase.upper() + ' total_loss: {losses.global_avg:.3f}'.format(losses=metric_logger.total_loss) + 
          ' hal_loss: {losses.global_avg:.3f}'.format(losses=metric_logger.hal_loss) + 
          ' sgt_loss: {losses.global_avg:.3f}'.format(losses=metric_logger.sgt_loss) )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Sigma pre-training scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()
    args.task = 'VLP'

    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)  

        args_log_path = output_path / "args.log"
        with open(args_log_path, 'w', encoding='utf-8') as f:
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
    main(args)
