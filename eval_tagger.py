import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from tqdm import tqdm

from datasets import TagDataset
from utils.metric import binary_accuracy

# Parameters
# folder with data files saved by create_input_files.py
data_folder = './scn_data'
# base name shared by data files
data_name = 'flickr10k_5_cap_per_img_5_min_word_freq'
# model checkpoint
checkpoint = './BEST_checkpoint_tagger_flickr10k_5_cap_per_img_5_min_word_freq.pth.tar'
# sets device for model and PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set to true only if inputs to model are fixed size; otherwise lot of computational overhead
cudnn.benchmark = True

# Load model
checkpoint = torch.load(checkpoint)
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate():
    """
    Evaluation

    Return
        Acc (float): score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        TagDataset(data_folder, data_name, 'TEST',
                   transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    accs = list()

    # For each image
    for i, (image, tags) in enumerate(tqdm(loader, desc="EVALUATING TAGS")):

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)
        targets = tags.to(device)  # (1, 1000)

        # Encode
        encoder_out = image  # (1, enc_image_size, enc_image_size, encoder_dim)
        scores = encoder(encoder_out)  # (1, 1000)
        acc = binary_accuracy(scores, targets)
        accs.append(acc)

    return sum(accs) / len(accs) if len(accs) > 0 else 0


if __name__ == '__main__':
    print("\nAccuracy score is %.4f." %
          (evaluate()))
