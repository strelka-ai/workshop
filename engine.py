import time
import torch

import torchvision.models.detection.mask_rcnn

import math
import sys
import utils
import numpy as np


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    tr_accuracy_list = []
    loss_classifier = []
    loss_box_reg = []
    loss_mask = []
    loss_aggregate = []

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        model.train()

        pure_masks = [t['masks'] for t in targets]

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

#         print(targets[0]['boxes'])

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            print(images)
            print(targets)
            sys.exit(1)

        try:
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        except:
            pass

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        current_accuracies_list = []
        with torch.no_grad():
            model.eval()
            outputs = model.forward(images)

            for i in range(0, len(pure_masks)):
                if outputs[i]['masks'].shape[0] == 0 and len(pure_masks) > 0:
                    current_accuracies_list.append(0)
                    continue

                masks1 = pure_masks[i]
                masks2 = sum((outputs[i]['masks'].chunk(chunks=outputs[i]['masks'].shape[0], dim=0))).squeeze(0).cpu()
                overlaps = utils.iou_masks(masks1, masks2)
                current_accuracies_list.append(overlaps)

        current_accuracy = 0 if len(current_accuracies_list) == 0 else float(np.mean(current_accuracies_list))

        tr_accuracy_list.append(current_accuracy)
        loss_classifier.append(float(loss_dict_reduced['loss_classifier'].cpu().item()))
        loss_box_reg.append(float(loss_dict_reduced['loss_box_reg'].cpu().item()))
        loss_mask.append(float(loss_dict_reduced['loss_mask'].cpu().item()))
        loss_aggregate.append(float(losses_reduced.cpu()))

    return {
        'tr_accuracy_list': tr_accuracy_list,
        'loss_classifier': loss_classifier,
        'loss_box_reg': loss_box_reg,
        'loss_mask':  loss_mask,
        'loss_aggregate': loss_aggregate
    }


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def iou_eval(box_a, box_b):
    # xA = max(box_a[0], box_b[0])
    # yA = max(box_a[1], box_b[1])
    # xB = min(box_a[2], box_b[2])
    # yB = min(box_a[3], box_b[3])
    # area_of_intersection = (xB - xA + 1) * (yB - yA + 1)

    xA = max(min(box_a[0], box_a[2]), min(box_b[0], box_b[2]))
    yA = max(min(box_a[1], box_a[3]), min(box_b[1], box_b[3]))
    xB = min(max(box_a[0], box_a[2]), max(box_b[0], box_b[2]))
    yB = min(max(box_a[1], box_a[3]), max(box_b[1], box_b[3]))

    if xA < xB and yA < yB:
        area_of_intersection = (xB - xA + 1) * (yB - yA + 1)
    else:
        area_of_intersection = 0

    box_a_area = (abs(box_a[2] - box_a[0] + 1)) * (abs(box_a[3] - box_a[1] + 1))
    box_b_area = (abs(box_b[2] - box_b[0] + 1)) * (abs(box_b[3] - box_b[1] + 1))

    iou = area_of_intersection / float(box_a_area + box_b_area - area_of_intersection)

    # if iou > 0:
    #     print(math.exp(iou) / math.exp(1))
    #     return iou

    return iou


def evaluate_item(target, output):
    bbox_accuracies = []
    
    if len(output['boxes']) == 0:
        return 0
    
    for t_bbox in target['boxes']:
        # current_accuracy = max([math.exp(iou_eval(t_bbox, o)) / math.exp(1) for o in output['boxes']])
        # current_accuracy = max([iou_eval(t_bbox, o) for o in output['boxes']])
        current_accuracy = 0
        for o in output['boxes']:
            if iou_eval(t_bbox, o):
                current_accuracy = 1
        # current_accuracy = max([iou_eval(t_bbox, o) for o in output['boxes']])
        bbox_accuracies.append(current_accuracy)

    if len(bbox_accuracies) == 0:
        return 0

    if (len(bbox_accuracies) - len(output['boxes'])) == 0:
        return 0

    return np.mean(bbox_accuracies)


def evaluate_tensor(targets, outputs):
    accuracies = []
    for i in range(0, len(targets)):
        accuracies.append(evaluate_item(targets[i], outputs[i]))

    return accuracies


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation:'

    accuracies = []
    i = 0
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        if device.type != 'cpu':
            torch.cuda.synchronize()
          
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        targets = [{k: v.to(cpu_device) for k, v in t.items()} for t in targets]

        for i in range(0, len(targets)):
            if outputs[i]['masks'].shape[0] == 0 and targets[i]['masks'].shape[0] > 0:
                accuracies.append(0)
                continue

            masks1 = targets[i]['masks']
            masks2 = sum((outputs[i]['masks'].chunk(chunks=outputs[i]['masks'].shape[0], dim=0))).squeeze(0).cpu()
            accuracy = utils.iou_masks(masks1, masks2)
            accuracies.append(accuracy)

        model_time = time.time() - model_time

        # res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        # utils.iou_masks(targets, outputs)
        # accuracies = evaluate_tensor(targets, outputs)

        mean_acc = float(np.mean(accuracies)) * 100

        evaluator_time = time.time()
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(accuracy=mean_acc, model_time=model_time, evaluator_time=evaluator_time)

        i += 1

    mean_acc = float(np.mean(accuracies))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('------')
    print("Усредненная статистика по валидации:", metric_logger)
#     print('Total avg accuracy: {}%'.format(mean_acc * 100))

    # accumulate predictions from all images
    torch.set_num_threads(n_threads)
    
    return accuracies

