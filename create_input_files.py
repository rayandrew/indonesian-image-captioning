from utils.dataset import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='flickr10k',
                       split_path='./dataset',
                       image_folder='./scn_dataset',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='./scn_data',
                       max_len=50)
