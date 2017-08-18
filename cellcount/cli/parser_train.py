from argparse import ArgumentDefaultsHelpFormatter


def func(args, parser):
    from os.path import join, isfile
    from collections import OrderedDict
    from glob import glob

    import torch
    import torch.optim as optim
    from torch import nn
    from torch.utils.data import DataLoader
    import torchvision.transforms as T

    import visdom

    from cellcount.utils import (ChunkSampler, ImageWithCount, train, test,
                                 get_val_example, push_epoch_image_count,
                                 save_checkpoint, reset)
    from cellcount.models import FPN, Counter
    from cellcount.losses import counter_loss

    if args.display:
        vis = visdom.Visdom(port=8080)

    BBBC = args.dataset
    NUM_TRAIN = 4000
    NUM_VAL = 1000
    BATCH_SIZE = args.batch_size
    gpu_dtype = torch.cuda.FloatTensor
    transform = T.Compose([T.Scale((256)),
                           T.RandomHorizontalFlip(),
                           T.ToTensor()])

    image_dir = glob(join(BBBC, '*images/'))[0]
    train_data = ImageWithCount(image_dir, transform=transform)

    loader_train = DataLoader(train_data, batch_size=BATCH_SIZE,
                              sampler=ChunkSampler(NUM_TRAIN, 0))
    loader_val = DataLoader(train_data, batch_size=BATCH_SIZE,
                            sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

    x_var, y_var = get_val_example(loader_val, gpu_dtype)
    _, _, h, w = x_var.size()

    fpn = FPN(h, w).type(gpu_dtype)
    reset(fpn)
    checkpoint = torch.load('fpn_model_best.pth.tar')
    fpn.load_state_dict(checkpoint['fpn'])

    count = Counter(h // 2, w // 2).type(gpu_dtype)
    model = nn.Sequential(OrderedDict([('fpn', fpn), ('counter', count)]))

    lr = args.learning_rate

    if args.cont and isfile('checkpoint.pth.tar'):
        print('Continuing from previous checkpoint...')
        checkpoint = torch.load('model_best.pth.tar')
        model.load_state_dict(checkpoint['model'])
        optimizer = optim.Adam(model.counter.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_loss = checkpoint['avg_val_loss']
    else:
        optimizer = optim.Adam(model.counter.parameters(), lr=lr)
        best_loss = 1E6

    epochs = args.num_epochs
    loss_fn = counter_loss
    val_loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        print('epoch: %s' % epoch)

        if epoch > 0 and (epoch % 20 == 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= .5

        train(loader_train, model, loss_fn, optimizer, gpu_dtype)
        val_loss = test(loader_val, model, val_loss_fn, gpu_dtype)
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
        save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'avg_val_loss': val_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best)

        if args.display:
            push_epoch_image_count(x_var, y_var, model, vis, epoch)


def configure_parser(sub_parsers):
    help = 'Train Counter'
    p = sub_parsers.add_parser('train', description=help, help=help,
                               formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset', type=str, help="Path to BBBC dataset",
                   required=True)
    p.add_argument('--num-epochs', type=int, help="Number of epochs",
                   default=1)
    p.add_argument('--batch-size', type=int, help="Batch size", default=5)
    p.add_argument('--learning-rate', type=float, help="Learning rate",
                   default=1E-4)
    p.add_argument('--cont', help="Continue from saved state",
                   action='store_true')
    p.add_argument('--display', help="Display via Visdom",
                   action='store_true')
    p.set_defaults(func=func)
