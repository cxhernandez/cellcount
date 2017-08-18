from argparse import ArgumentDefaultsHelpFormatter

DEFAULTS = {
    "bbbc005": {
        "images": "https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_images.zip",
        "truth": "https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_ground_truth.zip",
    }
}


def func(args, parser):
    import os
    import sys
    import urllib.request
    import tempfile
    from progressbar import (ProgressBar, Percentage, Bar,
                             ETA, FileTransferSpeed)

    dataset, outdir = args.dataset, args.outdir

    if dataset and dataset.lower() in DEFAULTS.keys():
        image_uri = DEFAULTS[dataset]['images']
        truth_uri = DEFAULTS[dataset]['truth']
        outdir = outdir or os.path.join('data', dataset)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    elif args.dataset not in DEFAULTS.keys():
        parser.error("Dataset %s unknown. Valid choices are: %s" %
                     (dataset, ", ".join(DEFAULTS.keys())))

    if image_uri is None:
        parser.error(
            "You must choose a known dataset.")
        sys.exit(1)
    if outdir is None:
        parser.error("You must provide an --outdir")
        sys.exit(1)

    image_fd = tempfile.NamedTemporaryFile()
    truth_fd = tempfile.NamedTemporaryFile()
    progress = ProgressBar(widgets=[Percentage(), ' ', Bar(), ' ', ETA(),
                                    ' ', FileTransferSpeed()])

    def update(count, blockSize, totalSize):
        if progress.max_value is None:
            progress.max_value = totalSize
            progress.start()
        progress.update(min(count * blockSize, totalSize))

    print('Downloading Dataset...')
    urllib.request.urlretrieve(image_uri, image_fd.name, reporthook=update)
    urllib.request.urlretrieve(truth_uri, truth_fd.name, reporthook=update)

    os.rename(image_fd.name, os.path.join(outdir, 'images.zip'))
    os.rename(truth_fd.name, os.path.join(outdir, 'truth.zip'))

    print("Done!")


def configure_parser(sub_parsers):
    help = 'Download BBBC datasets'
    p = sub_parsers.add_parser('download', description=help, help=help,
                               formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset', type=str,
                   help="Dataset name", choices=DEFAULTS.keys(), required=True)
    p.add_argument('-o', '--outdir', type=str, help='Output directory name')
    p.set_defaults(func=func)
