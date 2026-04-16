import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from models.models import Uni_Sign
import utils as utils
from datasets import S2T_Dataset
import os
import time
import argparse, json, datetime
from pathlib import Path
import math
import sys
from timm.optim import create_optimizer
from models.models import get_requires_grad_dict
from SLRT_metrics import translation_performance, islr_performance, wer_list
from transformers import get_scheduler
from config import *
from collections import OrderedDict
import random

def compute_batch_cka(stack_out):
    vis = stack_out['vis_features']    # [B, T_vis, D]
    txt = stack_out['txt_features']    # [B, T_txt, D]

    vis_pooled = vis.mean(dim=1)       # [B, D]
    txt_pooled = txt.mean(dim=1)       # [B, D]

    return linear_cka(vis_pooled, txt_pooled)

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    X: [N, D1]
    Y: [N, D2]
    return CKA 
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    numerator = (X.T @ Y).pow(2).sum()
    denom = torch.sqrt((X.T @ X).pow(2).sum() * (Y.T @ Y).pow(2).sum())

    return (numerator / denom).item()

def main(args):
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
    
    if args.dataset != 'How2Sign':
        dev_data = S2T_Dataset(path=dev_label_paths[args.dataset], 
                            args=args, phase='dev')
        print(dev_data)
        dev_sampler = torch.utils.data.SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers, 
                                    collate_fn=dev_data.collate_fn,
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
    model = Uni_Sign(args=args)
    model.to(device)
    model.train()
    for _, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')

    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=optimizer,
                num_warmup_steps=int(args.warmup_epochs * len(train_dataloader)/args.gradient_accumulation_steps),
                num_training_steps=int(args.epochs * len(train_dataloader)/args.gradient_accumulation_steps),
            )
    
    if args.use_deepspeed:
        model, optimizer, lr_scheduler = utils.init_deepspeed(args, model, optimizer, lr_scheduler)
        model_without_ddp = model.module.module
    print(optimizer)

    module = 'module.' if args.distributed else ''
    if args.finetune != '':
        print('***********************************')
        print(f'Load Checkpoint {args.finetune} ...')
        print('***********************************')
        state_dict = torch.load(args.finetune, map_location='cpu')['model']
        new_state_dict = OrderedDict()
        vlp_special = ['sgt_dec', 'stm', 'hcl']
        if args.finetune_w_vlp:
            for k, v in state_dict.items():
                if not any(special in k for special in vlp_special):
                    new_k = module + k
                    new_state_dict[new_k] = v
                if 'early_fusion_encoder.vis_encrd' in k:
                    if 'embed_tokens' in k:
                        new_k = module + 'mt5_model.shared.' + '.'.join(k.split('.')[4:])
                        new_state_dict[new_k] = v
                    new_k = module + 'mt5_model.encoder.' + '.'.join(k.split('.')[3:])
                    new_state_dict[new_k] = v
                if 'sgt_dec' in k :
                    part = 'decoder.'
                    if 'sgt_dec_lm_head' in k:
                        part = 'lm_head.'
                    new_k = module + 'mt5_model.' + part + '.'.join(k.split('.')[1:])
                    new_state_dict[new_k] = v
            ret = model.load_state_dict(new_state_dict, strict=False)
        else:
            ret = model.load_state_dict(state_dict, strict=False)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))

    output_dir = Path(args.output_dir)

    start_time = time.time()
    max_accuracy = 0
    if args.task == "CSLR":
        max_accuracy = 1000
    

    if args.resume:
        print(f'Resuming Model Parameters {args.resume}... ')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        if 'checkpoint' in args.resume:
            args.start_epoch = int(args.resume.split('/')[-1].split('.')[0].split('_')[-1])+1

    if args.eval:
        if utils.is_main_process():
            partition = ''
            if args.task != "ISLR" and args.dataset != 'How2Sign':
                print("DEV result")
                stats = evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev')
                stats_lst = [(k, v) for k, v in stats.items()]
                partition += ' DEV '
            if args.get_test_result or args.task == "ISLR" or args.dataset == 'How2Sign':
                print("TEST result")
                stats = evaluate(args, test_dataloader, model, model_without_ddp, phase='test')
                stats_lst = [(k, v) for k, v in stats.items()]
                partition += ' TEST '

        log_stats = {'partition': partition,
                     'stats': str(stats_lst),
                     'n_parameters': n_parameters}
        
        if args.output_dir:
            with (output_dir / "eval_out.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return 
    print(f"Start training for {args.epochs} epochs")

    save_epochs = args.save_epochs_lst

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(args, model, train_dataloader, optimizer, epoch)

        if args.save_some_checkpoint and (epoch+1) in save_epochs:
            if args.output_dir:
                checkpoint_paths = [output_dir / f'checkpoint_{epoch}.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': get_requires_grad_dict(model_without_ddp),
                    }, checkpoint_path)
        elif (epoch+1) == args.epochs:
            if args.output_dir:
                checkpoint_paths = [output_dir / f'checkpoint_{epoch}.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': get_requires_grad_dict(model_without_ddp),
                    }, checkpoint_path)

        # single gpu inference
        dev_stats, test_stats, stats = None, None, None
        if utils.is_main_process():
            if args.dataset != 'How2Sign':
                dev_stats = evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev')
            if args.get_test_result or epoch == args.epochs - 1: 
                test_stats = evaluate(args, test_dataloader, model, model_without_ddp, phase='test')

            if dev_stats is None and test_stats is None:
                raise NotImplementedError
            elif dev_stats is not None: 
                stats = dev_stats
                the_set = 'DEV'
            elif test_stats is not None:
                stats = test_stats
                the_set = 'TEST'

            if args.task == "SLT":
                if max_accuracy < stats["bleu4"]:
                    max_accuracy = stats["bleu4"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': get_requires_grad_dict(model_without_ddp),
                            }, checkpoint_path)

                if the_set == 'DEV':
                    print(f"[DEV] BLEU-4 of the network on the {dev_data} dev videos: {dev_stats['bleu4']:.2f}")
                elif the_set == 'TEST':
                    print(f"[TEST] BLEU-4 of the network on the {test_data} dev videos: {test_stats['bleu4']:.2f}")

                print(f'Max BLEU-4: {max_accuracy:.2f}%')
                
                if epoch == args.epochs - 1 and test_stats is not None:
                    print(f"[TEST] BLEU-4 of the network on the {test_data} test videos: {test_stats['bleu4']:.2f}")

            elif args.task == "ISLR":
                if max_accuracy < stats["top1_acc_pi"]:
                    max_accuracy = stats["top1_acc_pi"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': get_requires_grad_dict(model_without_ddp),
                            }, checkpoint_path)

                if the_set == 'DEV':
                    print(f"[DEV] PI accuracy of the network on the {dev_data} dev videos: {dev_stats['top1_acc_pi']:.2f}")
                elif the_set == 'TEST':
                    print(f"[TEST] PI accuracy of the network on the {test_data} dev videos: {test_stats['top1_acc_pi']:.2f}")
                
                print(f'Max PI accuracy: {max_accuracy:.2f}%')

                if epoch == args.epochs - 1 and test_stats is not None:
                    print(f"[TEST] PI accuracy of the network on the {test_data} test videos: {test_stats['top1_acc_pi']:.2f}")

            elif args.task == "CSLR":
                if max_accuracy > dev_stats["wer"]:
                    max_accuracy = dev_stats["wer"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': get_requires_grad_dict(model_without_ddp),
                            }, checkpoint_path)

                if the_set == 'DEV':
                    print(f"[DEV] WER of the network on the {dev_data} dev videos: {dev_stats['wer']:.2f}")
                elif the_set == 'TEST':
                    print(f"[TEST] WER of the network on the {test_data} dev videos: {test_stats['wer']:.2f}")
                
                print(f'Min WER: {max_accuracy:.2f}%')

                if epoch == args.epochs - 1 and test_stats is not None:
                    print(f"[TEST] WER of the network on the {test_data} test videos: {test_stats['wer']:.2f}")

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k} ({the_set.lower()})': v for k, v in dev_stats.items()},
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
    print('Done')
    
def train_one_epoch(args, model, data_loader, optimizer, epoch, dataset_name=None, steps=None):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    optimizer.zero_grad()

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):

        if dataset_name and steps:
            if not step in steps:
                metric_logger.update(loss=-1)
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                continue

        for key in src_input.keys():
            if isinstance(src_input[key], torch.Tensor):
                src_input[key] = src_input[key].to(torch.float32).cuda()

        if args.task == "CSLR":
            tgt_input['gt_sentence'] = tgt_input['gt_gloss']
        stack_out = model(src_input, tgt_input)
        
        total_loss = stack_out['loss']

        if args.use_deepspeed:
            model.backward(total_loss)
        else:
            total_loss.backward()

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print(f"Warning: Loss contains {total_loss.item()}! Skipping this batch.")
            optimizer.zero_grad()  
            continue  

        if args.use_deepspeed:
            model.step()
        else:
            optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if args.debug:
            break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(args, data_loader, model, model_without_ddp, phase):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'{phase.upper()}:'
    all_cka = []

    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []
 
        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            for key in src_input.keys():
                if isinstance(src_input[key], torch.Tensor):
                    src_input[key] = src_input[key].to(torch.float32).cuda()

            if args.task == "CSLR":
                tgt_input['gt_sentence'] = tgt_input['gt_gloss']
            stack_out = model(src_input, tgt_input)
            
            total_loss = stack_out['loss']
            metric_logger.update(loss=total_loss.item())
        
            output = model_without_ddp.generate(stack_out, 
                                                max_new_tokens=100, 
                                                num_beams = 4
                        )
            if args.get_cka and args.get_features:
                cka_val = compute_batch_cka(stack_out)
                all_cka.append(cka_val)                

            for i in range(len(output)):
                tgt_pres.append(output[i])
                tgt_refs.append(tgt_input['gt_sentence'][i])
            
            if args.debug:
                break
    
    if args.get_cka and args.get_features:
        print('Average CKA is', str(sum(all_cka) / len(all_cka)))

    tokenizer = model_without_ddp.mt5_tokenizer
    padding_value = tokenizer.eos_token_id
    
    pad_tensor = torch.ones(150-len(tgt_pres[0])).cuda() * padding_value
    tgt_pres[0] = torch.cat((tgt_pres[0],pad_tensor.long()),dim = 0)

    tgt_pres = pad_sequence(tgt_pres,batch_first=True,padding_value=padding_value)
    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)

    if args.dataset == 'CSL_Daily' and args.task == "SLT":
        tgt_pres = [' '.join(list(r.replace(" ",'').replace("\n",''))) for r in tgt_pres]
        tgt_refs = [' '.join(list(r.replace("，", ',').replace("？","?").replace(" ",''))) for r in tgt_refs]

    if args.task == "SLT":
        bleu_dict, rouge_score = translation_performance(tgt_refs, tgt_pres)
        for k,v in bleu_dict.items():
            metric_logger.meters[k].update(v)
        metric_logger.meters['rouge'].update(rouge_score)
    
    elif args.task == "ISLR":
        top1_acc_pi, top1_acc_pc = islr_performance(tgt_refs, tgt_pres)
        metric_logger.meters['top1_acc_pi'].update(top1_acc_pi)
        metric_logger.meters['top1_acc_pc'].update(top1_acc_pc)
        
    elif args.task == "CSLR":
        wer_results = wer_list(hypotheses=tgt_pres, references=tgt_refs)
        print(wer_results)
        for k,v in wer_results.items():
            metric_logger.meters[k].update(v)
    
    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        if args.output_dir != '':
            with open(args.output_dir+f'/{phase}_tmp_pres.txt','w') as f:
                for i in range(len(tgt_pres)):
                    f.write(tgt_pres[i]+'\n')
            with open(args.output_dir+f'/{phase}_tmp_refs.txt','w') as f:
                for i in range(len(tgt_refs)):
                    f.write(tgt_refs[i]+'\n')
            
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Sigma finetune scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()
    
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)  

        args_log_path = output_path / "args.log"
        with open(args_log_path, 'w', encoding='utf-8') as f:
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
    main(args)