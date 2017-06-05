from argparse import ArgumentDefaultsHelpFormatter


def func(args, parser):

        from os.path import join, isfile
        from collections import OrderedDict

        import torch
        import torch.optim as optim
        from torch import nn
        from torch.utils.data import DataLoader
        import torchvision.transforms as T

        from torch.autograd import Variable

        import visdom

        from cellcount.utils import (ChunkSampler, ImageWithCount, train, test,
                                     get_val_example, push_epoch_image_count,
                                     save_checkpoint, reset)
        from cellcount.models import FPN, Counter
        from cellcount.losses import counter_loss

        vis = visdom.Visdom(port=8080)

        BBBC = args.dataset
        NUM_TRAIN = 4000
        NUM_VAL = 1000
        BATCH_SIZE = args.batch_size
        gpu_dtype = torch.cuda.FloatTensor
        transform = T.Compose([T.Scale((256)), T.RandomHorizontalFlip(), T.ToTensor()])

        train_data = ImageWithCount(join(BBBC, 'BBBC005_v1_images/'),
                                    transform=transform)

        loader_val = DataLoader(train_data, batch_size=BATCH_SIZE,
                                sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

        x_var, y_var = get_val_example(loader_val, gpu_dtype)
        _, _, h, w = x_var.size()

        fpn = FPN(h, w).type(gpu_dtype)
        count = Counter(h // 2, w // 2).type(gpu_dtype)
        model = nn.Sequential(OrderedDict([('fpn', fpn), ('counter', count)]))

        print('Continuing from previous checkpoint...')
        checkpoint = torch.load('model_best.pth.tar')
        model.load_state_dict(checkpoint['model'])

        model.eval()

        for t, (x, y) in enumerate(loader_val):
            x_var = Variable(x.type(gpu_dtype))
            y_var = Variable(y.type(gpu_dtype))

            push_epoch_image_count(x_var, y_var, model, vis, t)

            if t == 10:
                break


def configure_parser(sub_parsers):
    help = 'Test Counter'
    p = sub_parsers.add_parser('test', description=help, help=help,
                               formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset', type=str, help="Path to BBBC dataset",
                   required=True)
    p.add_argument('--num-epochs', type=int, help="Number of epochs",
                   default=1)
    p.add_argument('--batch-size', type=int, help="Batch size", default=5)
    p.add_argument('--display', help="Display via Visdom",
                   action='store_true')
    p.set_defaults(func=func)
