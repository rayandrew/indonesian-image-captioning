import argparse

from utils.dataset import create_input_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='[Indonesian Image Captioning] -- Create Input Files')

    parser.add_argument('--dataset', '-d', help='type of dataset')
    parser.add_argument(
        '--split_path', '-s', help='split path (karpathy)')
    parser.add_argument(
        '--image_folder', '-if', help='path to image folder')
    parser.add_argument(
        '--output_folder', '-of', help='path to output folder')
    parser.add_argument('--captions_per_image', '-cpi', default=5,
                        type=int, help='captions per image')
    parser.add_argument('--min_word_freq', '-mwf', default=5,
                        type=int, help='min word freq')
    parser.add_argument('--max_len', '-ml', default=50,
                        type=int, help='max caps len')

    args = parser.parse_args()

    print('Creating input files...')

    # Create input files (along with word map)
    create_input_files(dataset=args.dataset,
                       split_path=args.split_path,
                       image_folder=args.image_folder,
                       captions_per_image=args.captions_per_image,
                       min_word_freq=args.min_word_freq,
                       output_folder=args.output_folder,
                       max_len=args.max_len)

    print('Input files created!')
