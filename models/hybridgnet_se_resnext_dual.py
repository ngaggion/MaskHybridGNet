import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import ChebConv, Pool, residualBlock
import torchvision.ops.roi_align as roi_align
from losses.diff_ras.polygon import SoftPolygon
import numpy as np

import warnings
warnings.filterwarnings("ignore")


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEResNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, base_width=4, reduction=16):
        super(SEResNeXtBlock, self).__init__()
        
        width = int(out_channels * (base_width / 64.)) * cardinality
        
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width, track_running_stats=False)
        
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(width, track_running_stats=False)
        
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, track_running_stats=False)
        
        self.se = SEBlock(out_channels, reduction)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, track_running_stats=False)
            )

    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        
        if self.shortcut is not None:
            residual = self.shortcut(x)
        
        out += residual
        out = F.relu(out)
        
        return out
    
class EncoderConv(nn.Module):
    def __init__(self, config):
        super(EncoderConv, self).__init__()
        self.latents = config['latents']
        self.c = config['initial_filters']

        maximum_amount_of_layers = int(np.log2(config['inputsize'])) - 2
        number_of_layers = min(maximum_amount_of_layers, 5)  

        self.hw = config['inputsize'] // (2**number_of_layers)
        self.filters = [self.c * 2**i for i in range(number_of_layers)]
        
        self.maxpool = nn.MaxPool2d(2)

        if config['raster_as_input']:
            input_channels = len(config['organs'])
        else:
            input_channels = 1

        # Create downsampling layers dynamically
        self.dconv_down = nn.ModuleList()
        for i in range(len(self.filters)):
            in_channels = input_channels if i == 0 else self.filters[i-1]
            out_channels = self.filters[i]
            self.dconv_down.append(SEResNeXtBlock(in_channels, out_channels))
        
        # Final convolutional layer
        self.dconv_final = SEResNeXtBlock(self.filters[-1], self.filters[-1])
        
        # Fully connected layers for mu and logvar
        final_conv_size = self.filters[-1] * self.hw * self.hw
        self.fc_mu = nn.Linear(in_features=final_conv_size, out_features=self.latents)
        self.fc_logvar = nn.Linear(in_features=final_conv_size, out_features=self.latents)

    def forward(self, x):
        conv_outputs = []
        
        for i, dconv in enumerate(self.dconv_down):
            x = dconv(x)
            conv_outputs.append(x)
            x = self.maxpool(x)
        
        x = self.dconv_final(x)
        
        x = x.view(x.size(0), -1)  # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        
        return x_mu, x_logvar, list(reversed(conv_outputs))


class DecoderConv(nn.Module):
    def __init__(self, config):
        super(DecoderConv, self).__init__()
        self.latents = config['latents']
        self.c = config['initial_filters']
        
        maximum_amount_of_layers = int(np.log2(config['inputsize'])) - 2
        number_of_layers = min(maximum_amount_of_layers, 5)
        
        self.hw = config['inputsize'] // (2**number_of_layers)

        self.filters = [self.c * 2**i for i in range(number_of_layers)]
        self.filters = self.filters + [self.filters[-1]]

        # Final output channels
        self.out_channels = len(config['organs'])
        
        # Initial linear layer to convert latent vector to feature maps
        self.fc = nn.Linear(self.latents, self.filters[-1] * self.hw * self.hw)
        
        # Create upsampling layers dynamically
        self.upconv = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        
        # Create transposed conv blocks (upsampling)
        for i, j in enumerate(range(len(self.filters)-1, 0, -1)):
            # Upsample from current decoder layer to next decoder layer
            self.upconv.append(nn.ConvTranspose2d(
                self.filters[j], self.filters[j-1], 
                kernel_size=2, stride=2, padding=0
            ))
            
            # Calculate combined channels after concatenation
            combined_channels = self.filters[j-1] * 2
            
            # Create the SEResNeXtBlock with the correct input channel count
            self.conv_blocks.append(SEResNeXtBlock(combined_channels, self.filters[j-1]))
        
        # Final convolution layer
        self.final_conv = nn.Conv2d(self.filters[0], self.out_channels, kernel_size=1)
        
    def forward(self, conv_outputs):
        # Reshape from latent space to initial feature map
        x = conv_outputs[0]
        
        # Store intermediate feature maps for graph decoder
        decoder_features = [x]
        
        # Upsampling path
        for i in range(1, len(self.filters)-1):
            # Upsample
            x = self.upconv[i](x)
            
            # Skip connections from encoder
            encoder_features = conv_outputs[i]
            
            # Concatenate along channel dimension
            x = torch.cat([x, encoder_features], dim=1)
            
            # Apply convolutional block directly with correct channel count
            x = self.conv_blocks[i](x)
            
            # Store features at each level for the graph decoder
            decoder_features.append(x)
        
        # Final convolution to get segmentation output
        segmentation = torch.sigmoid(self.final_conv(x))
        
        return segmentation, decoder_features

class SkipBlock(nn.Module):
    def __init__(self, in_filters, window):
        super(SkipBlock, self).__init__()
        
        self.window = window
        self.graphConv_pre = ChebConv(in_filters, 2, 1, bias = False) 
    
    def lookup(self, pos, layer, output_size = (1,1)):
        B = pos.shape[0]
        N = pos.shape[1]
        F = layer.shape[1]
        h = layer.shape[-1]
        
        ## Scale from [0,1] to [0, h]
        pos = pos * h
        
        _x1 = (self.window[0] // 2) * 1.0
        _x2 = (self.window[0] // 2 + 1) * 1.0
        _y1 = (self.window[1] // 2) * 1.0
        _y2 = (self.window[1] // 2 + 1) * 1.0
        
        boxes = []
        for batch in range(0, B):
            x1 = pos[batch,:,0].reshape(-1, 1) - _x1
            x2 = pos[batch,:,0].reshape(-1, 1) + _x2
            y1 = pos[batch,:,1].reshape(-1, 1) - _y1
            y2 = pos[batch,:,1].reshape(-1, 1) + _y2
            
            aux = torch.cat([x1, y1, x2, y2], axis = 1)            
            boxes.append(aux)
        
        skip = roi_align(layer, boxes, output_size = output_size, aligned=True)
        vista = skip.view([B, N, -1])

        return vista

    def forward(self, x, adj, layer):
        pos = self.graphConv_pre(x, adj)
        pos = torch.clip(pos, 0, 1)
        skip = self.lookup(pos, layer)
        return torch.cat([x, pos, skip], axis = 2), pos

class HybridDual(nn.Module):
    def __init__(self, config, downsample_matrices, upsample_matrices, adjacency_matrices):
        super(HybridDual, self).__init__()
        
        self.config = config
        self.z = config['latents']
        self.eval_sampling = False
        
        # Initialize encoder
        self.encoder = EncoderConv(config)
        
        # Initialize convolutional decoder
        self.decoder_conv = DecoderConv(config)
        
        # Initialize graph decoder components
        hw = config['inputsize'] // 2 ** len(config['filters'])
        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
                
        self.n_nodes = config['n_nodes']
        self.filters = config['filters']
        self.K = 6
        self.window = (3,3)
        
        # Graph decoder fully connected layer
        outshape = self.filters[-1] * self.n_nodes[-1]          
        self.dec_lin = nn.Linear(self.z, outshape)
        
        # Dynamic block creation for graph decoder
        # Estimate the number of feature maps after each skip connection
        # Now these will come from the convolutional decoder
        # Keep last N features for graph decoder

        decoder_features = self.decoder_conv.filters[:len(upsample_matrices)+1][::-1]
        skip_values = [0] + [2 + decoder_features[i] for i in range(0, len(upsample_matrices)+1)]

        self.blocks = nn.ModuleList()
        for i in range(len(upsample_matrices)+1):
            block = nn.ModuleList([
                ChebConv(self.filters[-(2*i+1)] + skip_values[i], self.filters[-(2*i+2)], self.K),
                nn.InstanceNorm1d(self.filters[-(2*i+2)]),
                ChebConv(self.filters[-(2*i+2)], self.filters[-(2*i+3)], self.K),
                nn.InstanceNorm1d(self.filters[-(2*i+3)]),
                SkipBlock(self.filters[-(2*i+3)], self.window)
            ])
            if i < len(upsample_matrices):  # Don't add skip connection and pool to the last block
                block.extend([
                    Pool()
                ])
            self.blocks.append(block)
        
        # Final convolution layers for graph decoder
        self.final_conv1 = ChebConv(self.filters[1] + skip_values[-1], self.filters[1], self.K)
        self.final_norm1 = nn.InstanceNorm1d(self.filters[1])
        self.final_conv2 = ChebConv(self.filters[1], self.filters[0], self.K, bias=False)
        
        self.rasterizer = SoftPolygon(inv_smoothness=0.1, mode="mask")
        
        self.reset_parameters()
    
    def save_checkpoint(self, path, epoch, iterations, optimizer):
        torch.save({
            'epoch': epoch,
            'iterations': iterations,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config
        }, path)

    def load_checkpoint(self, path, device):
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
    
    def reset_parameters(self):
        nn.init.normal_(self.dec_lin.weight, 0, 0.5)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var).clamp(min=1e-6)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) 
    
    def encode(self, x):
        mu, log_var, conv_outputs = self.encoder(x)
        return mu, log_var, conv_outputs
    
    def decode(self, z, conv_outputs):
        # First pass through convolutional decoder
        segmentation, decoder_features = self.decoder_conv(conv_outputs)

        decoder_features = decoder_features[-len(self.blocks):]

        # Now process through graph decoder using features from conv decoder
        x_graph = F.relu(self.dec_lin(z))
        x_graph = x_graph.reshape(x_graph.shape[0], -1, self.filters[-1])
        
        positions = []
        for i, block in enumerate(self.blocks):
            x_graph = F.relu(block[1](block[0](x_graph, self.adjacency_matrices[-(i+2)]._indices())))
            x_graph = F.relu(block[3](block[2](x_graph, self.adjacency_matrices[-(i+2)]._indices())))
            
            x_graph, pos = block[4](x_graph, self.adjacency_matrices[-(i+2)]._indices(), decoder_features[i])
            
            positions.append(pos)
            
            if len(block) > 5:  # If the block has pool
                x_graph = block[5](x_graph, self.upsample_matrices[-(i+1)])

        # Final convolutions
        x_graph = F.relu(self.final_norm1(self.final_conv1(x_graph, self.adjacency_matrices[0]._indices())))
        x_graph = self.final_conv2(x_graph, self.adjacency_matrices[0]._indices())
        
        return x_graph, positions[::-1], segmentation
        
    
    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.dim()}D")
        
        # Encode input
        self.mu, self.log_var, conv_outputs = self.encode(x)
        
        # Sample from latent space if in training mode
        z = self.sampling(self.mu, self.log_var) if (self.training or not self.eval_sampling) else self.mu

        x_graph, positions, segmentation = self.decode(z, conv_outputs)

        return x_graph, positions, segmentation
    
    def raster_independent(self, x, organs, resolution=512):
        output = []

        B = x.shape[0]
        
        for batch in range(B):
            for organ in np.unique(organs):
                landmarks_organ = x[batch, organs == organ].unsqueeze(0)
                if landmarks_organ.numel() == 0:
                    continue
                
                output.append(landmarks_organ)
        
        # Make a sanity check, if landmarks are not batcheable, the rasterizer should do it organ by organ
        try:
            landmarks_organ = torch.cat(output, axis=0)
            raster = self.rasterizer(landmarks_organ * resolution, resolution, resolution, 0.1)
            raster = torch.nan_to_num(raster, nan=0.0).clamp(0, 1)
            raster = raster.reshape(B, -1, resolution, resolution)
        except:
            raster = []
            for data in output:
                landmarks_organ = data     
                raster_organ = self.rasterizer(landmarks_organ * resolution, resolution, resolution, 0.1)
                raster_organ = torch.nan_to_num(raster_organ, nan=0.0).clamp(0, 1)
                raster.append(raster_organ)
            
            raster = torch.cat(raster, axis=0)
            raster = raster.reshape(B, -1, resolution, resolution)
        return raster

    # Backward-compatible alias for the legacy "naive" representation.
    def raster_naive(self, x, organs, resolution=512):
        return self.raster_independent(x, organs, resolution)

    # Backward-compatible alias for the legacy "non-naive" representation.
    def raster_non_naive(self, x, organs, circ_organ_order, resolution=512):
        return self.raster_unified(x, organs, circ_organ_order, resolution)

    def raster_unified(self, x, organs, circ_organ_order, resolution=512):
        output = []
        B = x.shape[0]
        
        # Extract all unique organ IDs from combined strings
        unique_organs = set()
        for org_str in organs:
            for org in str(org_str).split('-'):
                if org:  # Skip empty strings
                    unique_organs.add(org)
        unique_organs = sorted(list(unique_organs))
        
        for batch in range(B):
            for organ in unique_organs:
                # Create mask for nodes that belong to this organ
                mask = torch.tensor(circ_organ_order[str(organ)], requires_grad=False).to(x.device)
                landmarks_organ = x[batch, mask].unsqueeze(0)
                
                if landmarks_organ.numel() == 0:
                    continue
                
                output.append(landmarks_organ)
                
        # Make a sanity check, if landmarks are not batcheable, the rasterizer should do it organ by organ
        try:
            landmarks_organ = torch.cat(output, axis=0)
            raster = self.rasterizer(landmarks_organ * resolution, resolution, resolution, 0.1)
            raster = torch.nan_to_num(raster, nan=0.0).clamp(0, 1)
            raster = raster.reshape(B, -1, resolution, resolution)
        except:
            raster = []
            for data in output:
                landmarks_organ = data     
                raster_organ = self.rasterizer(landmarks_organ * resolution, resolution, resolution, 0.1)
                raster_organ = torch.nan_to_num(raster_organ, nan=0.0).clamp(0, 1)
                raster.append(raster_organ)
            
            raster = torch.cat(raster, axis=0)
            raster = raster.reshape(B, -1, resolution, resolution)
        return raster