from argparse import ArgumentDefaultsHelpFormatter


def func(args, parser):
    from os.path import basename, join, isfile

    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    import visdom

    from cellcount.utils import (ChunkSampler, ImageWithMask, train, test,
                                 get_val_example, push_epoch_image,
                                 save_checkpoint)
    from cellcount.models import FPN
    from cellcount.losses import fpn_loss

    if args.display:
        vis = visdom.Visdom(port=8080)

    BBBC = args.dataset
    NUM_TRAIN = 480
    NUM_VAL = 120
    BATCH_SIZE = args.batch_size
    gpu_dtype = torch.cuda.FloatTensor

    train_data = ImageWithMask(join(BBBC, 'BBBC005_v1_ground_truth/'))
    train_data.imgs = [(join(BBBC, 'BBBC005_v1_images/jpg/%s') % basename(i), i)
                       for i, _ in train_data.imgs]

    loader_train = DataLoader(train_data, batch_size=BATCH_SIZE,
                              sampler=ChunkSampler(NUM_TRAIN, 0))
    loader_val = DataLoader(train_data, batch_size=BATCH_SIZE,
                            sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

    x_var, y_var = get_val_example(loader_val, gpu_dtype)
    _, _, h, w = x_var.size()

    fpn = FPN(h, w).type(gpu_dtype)
    lr = args.learning_rate

    if args.cont and isfile('fpn_checkpoint.pth.tar'):
        print('Continuing from previous checkpoint...')
        checkpoint = torch.load('fpn_model_best.pth.tar')
        fpn.load_state_dict(checkpoint['encoder'])
        optimizer = optim.Adam(fpn.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_loss = checkpoint['avg_val_loss']
    else:
        optimizer = optimizer = optim.Adam(fpn.parameters(), lr=lr)
        best_loss = 1E6

    epochs = args.num_epochs
    for epoch in range(epochs):
        print('epoch: %s' % epoch)

        if epoch > 0 and (epoch % 20 == 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= .5

        train(loader_train, fpn, fpn_loss, optimizer, gpu_dtype)
        val_loss = test(loader_val, fpn, fpn_loss, gpu_dtype)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss

        save_checkpoint({
            'epoch': epoch,
            'fpn': fpn.state_dict(),
            'avg_val_loss': val_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, name='fpn')

        if args.display:
            push_epoch_image(x_var, y_var, fpn, vis, epoch)


def configure_parser(sub_parsers):
    help = 'Train FPN'
    p = sub_parsers.add_parser('train_fpn', description=help, help=help,
                               formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset', type=str, help="Path to BBBC dataset",
                   required=True)
    p.add_argument('--num-epochs', type=int, help="Number of epochs",
                   default=1)
    p.add_argument('--batch-size', type=int, help="Batch size", default=5)
    p.add_argument('--learning-rate', type=float, help="Learning rate",
                   default=1E-3)
    p.add_argument('--cont', help="Continue from saved state",
                   action='store_true')
    p.add_argument('--display', help="Continue from saved state",
                   action='store_true')
    p.set_defaults(func=func)
