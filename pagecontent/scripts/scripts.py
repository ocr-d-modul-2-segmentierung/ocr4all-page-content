import argparse
import glob
from pagecontent.detection.detection import PageContentDetection, PageContentSettings
import tqdm


def glob_all(filenames):
    files = []
    for f in filenames:
        files += glob.glob(f)

    return files


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', ''):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description='Detects music lines in historical documents')
    parser.add_argument("--load", type=str,
                        help="Model to load")
    parser.add_argument("--space_height", type=int, default=20,
                        help="Average space between two lines. If set to '0',"
                             " the value will be calculated automatically ...")
    parser.add_argument("--target_line_height", type=int, default=6,
                        help="Scale the data images so that the space height"
                             " matches this value (must be the same as in training)")
    parser.add_argument("--binary", type=str, required=True, nargs="+",
                        help="directory name of the grayscale images")
    parser.add_argument("--processes", type=int, default=8,
                        help="Number of processes to use")
    parser.add_argument("--debug", type=str2bool, default=False,
                        help="Display debug images")
    args = parser.parse_args()

    binary_file_paths = sorted(glob_all(args.binary))
    print("Loading {} files with character height {}".format(len(binary_file_paths), args.space_height))

    settings = PageContentSettings(
        debug=args.debug,
        line_space_height=args.space_height,
        target_line_space_height=args.target_line_height,
        model=args.load,
        processes=args.processes,

    )
    lineDetector = PageContentDetection(settings)
    for i_ind, i in tqdm.tqdm(enumerate(lineDetector.detect_paths(binary_file_paths)), total=len(binary_file_paths)):
        pass


if __name__ == "__main__":
    main()
