import math
import torch
import helper

def kd_train_one_epoch(epoch, num_epochs,
                       student, teacher, teacher_without_ddp, defined_loss, data_loader,
                       optimizer, lr_schedule, wd_schedule, momentum_schedule,
                       clip_grad, freeze_last_layer, fp16_scaler):
    metric_logger = helper.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, num_epochs)
    # if helper.is_main_process():
    #     for print_epoch in range(10):
    #         print(f'At epoch {print_epoch}, the learning rate should be {lr_schedule[print_epoch*len(data_loader)]} according to the scheduler.')
    #     print()
    # optimizer.param_groups starts with [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]
    # optimizer.param_groups[0].keys(): dict_keys(['params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad'])
    # optimizer.param_groups[1].keys(): dict_keys(['params', 'weight_decay', 'lr', 'betas', 'eps', 'amsgrad'])

    for it, (images, labels) in enumerate(metric_logger.log_every(iterable=data_loader, print_freq=20, header=header)):
        # Update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # Move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # Teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # Each rank has size of teacher and student outputs:
            # torch.Size([BATCH_SIZE/NUM_GPUS*NUM_GLOBAL_CROPS, OUT_DIM]),
            # torch.Size([BATCH_SIZE/NUM_GPUS*NUM_ALL_CROPS, OUT_DIM])
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = defined_loss(student_output, teacher_output, epoch)
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # Update student network's parameters
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if clip_grad:
                param_norms = helper.clip_gradients(student, clip_grad)
            helper.cancel_gradients_last_layer(epoch, student, freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = helper.clip_gradients(student, clip_grad)
            helper.cancel_gradients_last_layer(epoch, student, freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # Teacher update with EMA 
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for student_ps, teacher_ps in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                teacher_ps.data.mul_(m).add_((1 - m) * student_ps.detach().data)

        # Logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    global_avg_stats =  {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    median_stats =  {k: meter.median for k, meter in metric_logger.meters.items()}
    avg_stats =  {k: meter.avg for k, meter in metric_logger.meters.items()}
    max_stats =  {k: meter.max for k, meter in metric_logger.meters.items()}
    value_stats =  {k: meter.value for k, meter in metric_logger.meters.items()}
    return global_avg_stats, median_stats, avg_stats, max_stats, value_stats

def get_optimizer(optimizer_choice, params_dict, lr, momentum):
    if optimizer_choice == "adamw":
        optimizer = torch.optim.AdamW(params_dict)  # to use with ViTs
    elif optimizer_choice == "sgd":
        optimizer = torch.optim.SGD(params_dict, lr=lr, momentum=momentum)  # lr is set by scheduler
    elif optimizer_choice == "lars":
        optimizer = helper.LARS(params_dict)  # to use with convnet and large batches
    return optimizer

def linear_train_one_epoch(model, linear_classifier, optimizer, data_loader, epoch, n, avgpool, model_params, mode='eval'):
    if mode=='train_finetuning':
        model.train()
    linear_classifier.train()
    metric_logger = helper.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', helper.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    if len(data_loader)<20:
        print_freq = 20
    else:
        print_freq = 20
    for (inp, target) in metric_logger.log_every(iterable=data_loader, print_freq=print_freq, header=header):
        # Move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if mode=='train_finetuning':
            if "vit" in model_params['backbone_option']:
                intermediate_output = model.module.backbone.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        elif mode=='eval':
            # Forward - we do not update the backbone
            with torch.no_grad():
                if "vit" in model_params['backbone_option']:
                    intermediate_output = model.module.backbone.get_intermediate_layers(inp, n)
                    output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                    if avgpool:
                        output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                        output = output.reshape(output.shape[0], -1)
                else:
                    output = model(inp)
        output = linear_classifier(output)

        # Compute cross entropy loss
        loss = torch.nn.CrossEntropyLoss()(output, target)

        # Compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # Step
        optimizer.step()

        # Log
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}