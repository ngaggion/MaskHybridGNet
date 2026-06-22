import argparse
import json
import numpy as np

from training.trainer_mse import trainer
from models.hybridgnet_se_resnext import Hybrid
from data.dataset import LandmarksDataset, ToTensorWithSegBatched as ToTensor
from data.transforms import Scale, AugColor, RandomScaleCentered, Rotate, RandomHorizontalFlip
from torchvision import transforms
from models.utils import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str) # Name of the experiment
    parser.add_argument("--dataset", type=str) # Path to the dataset
    parser.add_argument("--resume", type=str, help="Path to a checkpoint to resume training")
    
    # Training parameters
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--warm_up_it", default=25000, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--stepsize", default=200000, type=int)
    parser.add_argument("--gamma", default=0.1, type=float)
    
    # Losses weights
    parser.add_argument("--mse_w", default=10.0, type=float)
    parser.add_argument("--kld_w", default=1e-4, type=float)
    parser.add_argument("--edge_w", default=1.0, type=float)
    parser.add_argument("--dice_w", default=2.0, type=float)
    parser.add_argument("--bce_w", default=1.0, type=float)

    # Model parameters
    parser.add_argument("--latents", default=64, type=int)
    parser.add_argument("--initial_filters", default=16, type=int)
    
    # Data augmentation
    parser.add_argument("--flip", type=bool, default=True)
    parser.add_argument("--no-flip", dest='flip', action='store_false')
    parser.add_argument("--rotate", type=bool, default=True)
    parser.add_argument("--no-rotate", dest='rotate', action='store_false')

    # Seg-to-Graph parameters
    parser.add_argument("--raster-as-input", type=bool, default=False)
    parser.add_argument("--raster-input", dest='raster_as_input', action='store_true')

    # Graph representation: independent (per-organ graphs) or unified (shared boundaries)
    parser.add_argument("--representation", choices=["independent", "unified"], default="independent")
    # Deprecated aliases kept for backward compatibility
    parser.add_argument("--naive", dest='representation', action='store_const', const='independent')
    parser.add_argument("--non-naive", dest='representation', action='store_const', const='unified')
    parser.add_argument("--nonnaive", dest='representation', action='store_const', const='unified')
    
    args = parser.parse_args()
    config = vars(args)

    DATASET = config['dataset']
    config, D_t, U_t, A_t = load_config(DATASET, config)

    # Creation info, read the config.json file and print the information
    data_config = json.load(open(f"{DATASET}/config.json"))
    for key, value in data_config.items():
        print(key, ":", value)

    images_train = np.loadtxt(f"{DATASET}/train.txt", dtype=str)
    images_val = np.loadtxt(f"{DATASET}/val.txt", dtype=str)
    print("Train images", len(images_train))
    print("Val images", len(images_val))
    print("")
    
    transforms_list = [Scale(config['inputsize']), RandomScaleCentered(config['inputsize'])]
    if config['rotate']:
        transforms_list.append(Rotate(45))
    if config['flip']:
        transforms_list.append(RandomHorizontalFlip())

    transforms_list.extend([AugColor(0.40), ToTensor()])
    
    train_dataset = LandmarksDataset(
        images_train, f"{DATASET}/images", f"{DATASET}/landmarks2",
        transform=transforms.Compose(transforms_list)
    )

    val_dataset = LandmarksDataset(
        images_val, f"{DATASET}/images", f"{DATASET}/landmarks2",
        transform=transforms.Compose([Scale(config['inputsize']), ToTensor()])
    )

    model = Hybrid(config, D_t, U_t, A_t)
        
    print('Model: HybridGNet')
    print("Image Encoder filters", model.encoder.filters + [model.encoder.filters[-1]])
    print("Bottleneck latents", model.encoder.latents)
    print("Graph convolutional filters", config['filters'][::-1])
    print("")
    
    if config['resume']:
        print(f"Loading checkpoint from {config['resume']}")
        checkpoint = model.load_checkpoint(config['resume'], config['device'])
        start_epoch = checkpoint['epoch'] + 1
        start_iterations = checkpoint['iterations'] + 1
        print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0
        start_iterations = 0
        checkpoint = None

    # Add the first resolution to the list of resolutions for deep supervision
    config['resolutions'] = [config['resolutions'][0]] + config['resolutions']

    trainer(train_dataset, val_dataset, model, config, start_epoch, start_iterations, checkpoint)