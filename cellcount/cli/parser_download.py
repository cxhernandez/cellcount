from argparse import ArgumentDefaultsHelpFormatter

BROAD_URL = "https://data.broadinstitute.org/bbbc/"
DEFAULTS = {
    "bbbc005": {
        "images":
            BROAD_URL + "BBBC005/BBBC005_v1_images.zip",
        "truth":
            BROAD_URL + "BBBC005/BBBC005_v1_ground_truth.zip",
    }
}


def func(args, parser):
    import os
    import sys
    import urllib.request
    import tempfile
    import zipfile
    from PIL import Image
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

    progress = ProgressBar(widgets=[Percentage(), ' ', Bar(), ' ', ETA(),
                                    ' ', FileTransferSpeed()])

    def update(count, blockSize, totalSize):
        if progress.max_value is None:
            progress.max_value = totalSize
            progress.start()
        progress.update(min(count * blockSize, totalSize))

    def dl_unzip(uri, outdir, reporthook):

        with tempfile.NamedTemporaryFile() as tmp:
            urllib.request.urlretrieve(uri, tmp.name,
                                       reporthook=reporthook)
            print("\nExtracting...")
            with zipfile.ZipFile(tmp.name, 'r') as zf:
                zf.extractall(outdir)

    def tif2jpg(data_path):
        for root, dirs, files in os.walk(data_path, topdown=False):
            for name in files:
                full_path = os.path.join(root, name)
                fn, ext = os.path.splitext(full_path)
                if ext.lower().startswith('.tif'):
                    if args.exclude:
                        if fn.find(args.exclude) == -1:
                            os.remove(full_path)
                            continue
                    if os.path.isfile(fn + ".jpg"):
                        print("A JPEG already exists for %s." % name)
                    else:
                        outfile = fn + ".jpg"
                        try:
                            im = Image.open(full_path)
                            print("Generating JPEG for %s..." % name)
                            im.thumbnail(im.size)
                            im.save(outfile, "JPEG", quality=100)
                            os.remove(full_path)

                        except Exception as e:
                            print(e)

    print("Downloading imageset...")
    dl_unzip(image_uri, outdir, update)

    print("Downloading ground truth set...")
    dl_unzip(truth_uri, outdir, update)

    print("Converting to JPEG...")
    tif2jpg(outdir)

    print("Done!")


def configure_parser(sub_parsers):
    help = 'Download BBBC datasets'
    p = sub_parsers.add_parser('download', description=help, help=help,
                               formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset', type=str,
                   help="Dataset name", choices=DEFAULTS.keys(), required=True)
    p.add_argument('-o', '--outdir', type=str, help='Output directory name')
    p.add_argument('-e', '--exclude', type=str, help='exclude')
    p.set_defaults(func=func)
