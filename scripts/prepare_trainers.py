import math
import torch
import helper

def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch, num_epochs, clip_grad, freeze_last_layer,
                    fp16_scaler):
    metric_logger = helper.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, num_epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(iterable=data_loader, print_freq=10, header=header)):
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
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)      # Each rank has size of teacher and student outputs:
                                                  # torch.Size([BATCH_SIZE/NUM_GPUS*NUM_GLOBAL_CROPS, OUT_DIM]),
                                                  # torch.Size([BATCH_SIZE/NUM_GPUS*NUM_ALL_CROPS, OUT_DIM])

            loss = dino_loss(student_output, teacher_output, epoch)
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

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}