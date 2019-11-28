import os
import sys
import cv2
import time
import torch
import torch.utils.data
import torchvision
import numpy as np
from get_model import get_model, network_models
from dataset import MaskDataset
from engine import train_one_epoch, evaluate
import utils
from utils import get_rank
import matplotlib.pyplot as plt


# chelyabinsk
# krasnoyarsk
# nnovgorod
# novosibirsk
# perm
# rnd
# samara
# ufa

def get_root():
  if getattr(sys, 'frozen', False):
    return os.path.dirname(sys.executable)
  elif __file__:
    return os.path.abspath(os.path.dirname(__file__))


def get_weights_filepath(epoch, dataset, network, is_train=False):

    file_segments = []

    if is_train:
        file_segments.append('train')

    if epoch is not None:
        file_segments.append('ep{}'.format(epoch))

    file_segments.append(dataset)
    file_segments.append(network)

    root = get_root()
    file_name = '_'.join(file_segments)
    return os.path.join(root, 'weights', file_name)


def main():

    DATASET = 'water'
    NETWORK = 'resnet'
    NUM_EPOCHS = 50
    EPOCH_SAVING_RATE = 5
    IS_FIRST_RUN = True

    root = get_root()

    if len(sys.argv) > 1:
        if sys.argv[1] in network_models.keys():
            NETWORK = sys.argv[1]
        else:
            print('Wrong model!')
            return 1

    if len(sys.argv) > 3:
        DATASET = sys.argv[2]
        if not os.path.isdir(os.path.join(root, 'datasets', DATASET)):
            print('Wrong dataset!')
            return 2

    if len(sys.argv) > 4:
        NUM_EPOCHS = int(sys.argv[3])

    weights_file = os.path.join('weights', DATASET + '_' + NETWORK + '_initial_weights.pt')
    train_weights_file = DATASET + '_' + 'train_weights.pt'

    # if os.path.isfile(weights_file) and os.path.getsize(weights_file) > 0:
    #     IS_FIRST_RUN = False

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    dataset = MaskDataset(os.path.join(root, 'datasets', DATASET, 'train'))
    dataset_test = MaskDataset(os.path.join(root, 'datasets', DATASET, 'test'))

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-1320])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[:10]) # -50:

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn)

    starting_ts = time.time()

    print('Starting the teaching process: {}'.format(time.asctime()))
    print('Network: {}'.format(NETWORK))
    print('Dataset: {}'.format(DATASET))
    print('Epochs: {}'.format(NUM_EPOCHS))
    print('Device: {}'.format( ('cuda' if torch.cuda.is_available() else 'cpu' ) ))

    if not IS_FIRST_RUN:
        print('Preload weights: {}'.format(weights_file))

    print('--- -- -- -- -- -- ---')

    model = get_model(NETWORK)

    if not IS_FIRST_RUN:
        model.load_state_dict(torch.load(weights_file))

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    learning_ts_list = list()
    for epoch in range(NUM_EPOCHS):
        epoch_learning_ts = time.time()

        try:
            # train for one epoch, printing every 10 iterations
            losses, accuracies = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

            # update the learning rate
            lr_scheduler.step()

            # evaluate on the test dataset
            accuracies = evaluate(model, data_loader_test, device=device)

            mean_acc = float(np.mean(accuracies))

            print('Epoch {} is done, mean accuracy {}%'.format(epoch, mean_acc * 100))

            if epoch >= EPOCH_SAVING_RATE and epoch % EPOCH_SAVING_RATE ==0:
                train_weights_file_path = get_weights_filepath(epoch=epoch, dataset=DATASET, network=NETWORK, is_train=True)
                torch.save(model.state_dict(), train_weights_file_path)

            print('place to plot')

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(e)

            # saving exist
            train_weights_file_path = get_weights_filepath(epoch=epoch, dataset=DATASET, network=NETWORK, is_train=True)
            torch.save(model.state_dict(), train_weights_file_path)

        epoch_learning_ts = time.time() - epoch_learning_ts
        learning_ts_list.append(epoch_learning_ts)
        avg_learn_time = np.mean(learning_ts_list)

        print('Learning time: {} sec'.format(int(epoch_learning_ts)))
        print('Avg learning time: {} sec'.format(int(avg_learn_time)))

    weights_file_path = get_weights_filepath(epoch=None, dataset=DATASET, network=NETWORK, is_train=False)
    torch.save(model.state_dict(), weights_file_path)

    print("That's it!")

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
