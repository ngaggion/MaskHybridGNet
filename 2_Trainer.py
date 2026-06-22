import argparse
import json
import numpy as np

from training.trainer import trainer
from training.pretrainer_mse import pretrainer
from models.hybridgnet_se_resnext_dual import HybridDual
from models.hybridgnet_se_resnext import Hybrid
from data.dataset import LandmarksDataset, ToTensorWithSegBatched as ToTensor
from data.transforms import Scale, AugColor, RandomScaleCentered, Rotate, RandomHorizontalFlip, RandomVerticalFlip, RandomTranspose
from torchvision import transforms
from models.utils import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str) # Name of the experiment
    parser.add_argument("--dataset", type=str) # Path to the dataset
    parser.add_argument("--resume", type=str, help="Path to a checkpoint to resume training")
    
    # Training parameters

    parser.add_argument("--iterations", default=250000, type=int)
    parser.add_argument("--epochs", default=0, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--warm_up_it", default=50000, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-6, type=float)
    parser.add_argument("--gamma", default=0.1, type=float)
    
    # Losses weights
    parser.add_argument("--chamfer_w", default=10.0, type=float)
    parser.add_argument("--kld_w", default=1e-3, type=float)
    parser.add_argument("--dice_w", default=1.0, type=float)
    parser.add_argument("--bce_w", default=1.0, type=float)
    parser.add_argument("--conv_dice_w", default=1.0, type=float)
    parser.add_argument("--conv_bce_w", default=1.0, type=float)
    parser.add_argument("--edge_w", default=1.0, type=float)
    parser.add_argument("--elasticity_w", default=250.0, type=float)
    parser.add_argument("--curvature_w", default=300.0, type=float)

    # Model parameters
    parser.add_argument("--latents", default=64, type=int)
    parser.add_argument("--initial_filters", default=16, type=int)
    
    # Data augmentation
    parser.add_argument("--flip_h", type=bool, default=True)
    parser.add_argument("--no-flip-h", dest='flip_h', action='store_false')
    parser.add_argument("--flip_v", type=bool, default=True)
    parser.add_argument("--no-flip-v", dest='flip_v', action='store_false')
    parser.add_argument("--transpose", type=bool, default=True)
    parser.add_argument("--no-transpose", dest='transpose', action='store_false')

    parser.add_argument("--rotate", type=bool, default=True)
    parser.add_argument("--no-rotate", dest='rotate', action='store_false')

    # Seg-to-Graph parameters
    parser.add_argument("--raster-as-input", type=bool, default=False)
    parser.add_argument("--raster-input", dest='raster_as_input', action='store_true')

    # Use Dual Model instead, default is False
    parser.add_argument("--use-dual", type=bool, default=False)
    parser.add_argument("--dual", dest='use_dual', action='store_true')
    
    # Use rasterization loss
    parser.add_argument("--use-raster", type=bool, default=True)
    parser.add_argument("--no-raster", dest='use_raster', action='store_false')
    
    # Graph representation: independent (per-organ graphs) or unified (shared boundaries)
    parser.add_argument("--representation", choices=["independent", "unified"], default="independent")
    # Deprecated aliases kept for backward compatibility
    parser.add_argument("--naive", dest='representation', action='store_const', const='independent')
    parser.add_argument("--non-naive", dest='representation', action='store_const', const='unified')
    parser.add_argument("--nonnaive", dest='representation', action='store_const', const='unified')
    
    # Validate every N epochs
    parser.add_argument("--val_every", default=1, type=int)
    
    # Resume as pretraining
    parser.add_argument("--pretraining", type=bool, default=False)
    parser.add_argument("--pretrain", dest='pretraining', action='store_true')
    
    # Production mode
    parser.add_argument("--production", type=bool, default=False)
    parser.add_argument("--prod", dest='production', action='store_true')

    args = parser.parse_args()
    config = vars(args)

    DATASET = config['dataset']
    config, D_t, U_t, A_t = load_config(DATASET, config)
    print(config)
    
    if not config['production']:
        images_train = np.loadtxt(f"{DATASET}/train.txt", dtype=str)
        images_val = np.loadtxt(f"{DATASET}/val.txt", dtype=str)
        print("Train images", len(images_train))
        print("Val images", len(images_val))
        print("")
    else:
        images_train = np.loadtxt(f"{DATASET}/train.txt", dtype=str)
        images_val = np.loadtxt(f"{DATASET}/val.txt", dtype=str)
        images_test = np.loadtxt(f"{DATASET}/test.txt", dtype=str)
        # we put all images together for training
        images_train = np.concatenate((images_train, images_val, images_test))

        print("Train images", len(images_train))
        print("Val images", len(images_val))
        print("")

    if config['epochs'] == 0:
        config['epochs'] = config['iterations'] // len(images_train)
        print(f"Setting epochs to {config['epochs']}")

    # There will be two steps in the learning rate schedule
    config['stepsize'] = config['epochs'] // 2
    
    transforms_list = [Scale(config['inputsize']), RandomScaleCentered(config['inputsize'])]
    if config['rotate']:
        transforms_list.append(Rotate(45))
        print("Using random rotation augmentation")
    if config['transpose']:
        transforms_list.append(RandomTranspose())
        print("Using random transpose augmentation")
    if config['flip_h']:
        transforms_list.append(RandomHorizontalFlip())
        print("Using random horizontal flip augmentation")
    if config['flip_v']:
        transforms_list.append(RandomVerticalFlip())
        print("Using random vertical flip augmentation")

    transforms_list.extend([AugColor(0.40), ToTensor()])
    
    train_dataset = LandmarksDataset(
        images_train, f"{DATASET}/images", f"{DATASET}/landmarks",
        transform=transforms.Compose(transforms_list)
    )

    val_dataset = LandmarksDataset(
        images_val, f"{DATASET}/images", f"{DATASET}/landmarks", 
        transform=transforms.Compose([Scale(config['inputsize']), ToTensor()])
    )

    if config['use_dual']:
        model = HybridDual(config, D_t, U_t, A_t)
        print('Model: HybridGNet Dual')
    else:
        model = Hybrid(config, D_t, U_t, A_t)
        print('Model: HybridGNet')
        
    # The regularization hyperparameters are set to the number of 3 resolutions by default
    # We found out that the best results are obtained with 3 resolutions
    # and that the regularization hyperparameters are a bit sensitive when going over 3 resolutions
    # as we regularize at each resolution, their amount in the loss function gets bigger
    # and we need to decrease them
    
    if len(config['resolutions']) > 3:
        config['elasticity_w'] = config['elasticity_w'] * 3 / len(config['resolutions'])
        config['curvature_w'] = config['curvature_w'] * 3 / len(config['resolutions'])
            
    print("Image Encoder filters", model.encoder.filters + [model.encoder.filters[-1]])
    print("Bottleneck latents", model.encoder.latents)
    print("Graph convolutional filters", config['filters'][::-1])
    print("")
    
    if config['resume']:
        print(f"Loading checkpoint from {config['resume']}")
        checkpoint = model.load_checkpoint(config['resume'], config['device'])
        
        if "PreTrain" in config["resume"] or config['pretraining']:
            print("PreTraining loaded")
            checkpoint["optimizer_state_dict"] = None
            checkpoint['epoch'] = 0
            checkpoint['iterations'] = 0
            
        start_epoch = checkpoint['epoch'] + 1
        start_iterations = checkpoint['iterations'] + 1
        print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}")
    elif config['pretraining']:
        print("Pretraining mode enabled, starting from scratch.")
        model = pretrainer(model, config)
        start_epoch = 0
        start_iterations = 0
        checkpoint = None
    else:
        start_epoch = 0
        start_iterations = 0
        checkpoint = None

    # Add the first resolution to the list of resolutions for deep supervision
    config['resolutions'] = [config['resolutions'][0]] + config['resolutions']

    trained_correctly = False
    trained_correctly = trainer(train_dataset, val_dataset, model, config, start_epoch, start_iterations, checkpoint, validate_every=config['val_every'])
    
    if trained_correctly:
        print("Training completed successfully.")
    else:
        print("Training did not complete successfully. Please check the logs for errors.")