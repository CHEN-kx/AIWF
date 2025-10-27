# AIWx
# We conducted the reproduction based on the pseudo-code provided by the official website 
# https://github.com/198808xc/Pangu-Weather
# <Accurate medium-range global weather forecasting with 3D neural networks>

import timm
import numpy as np

import torch
from torch import nn
from torchvision import transforms
from torch import arange as RangeTensor

# modified from https://github.com/microsoft/Swin-Transformer
def window_partition(x:torch.Tensor, window_size):
    """
    Args:
        x: (B,Z, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, Z, H, W, C = x.shape # [1,8,186,360,1]
    x = x.view(B,Z//window_size[0],window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C) #[1,4,2,31,6,30,12,1]
    windows = x.permute(0, 1, 3, 5, 2, 4,6,7) #.contiguous().view(-1, window_size[0], window_size[1],window_size[2], C) # [1,4,2,31,6,30,12,1]->[1,4,31,30,2,6,12,1]
    return windows

# modified from https://github.com/microsoft/Swin-Transformer
def gen_mask(x_shape,window_size) -> torch.Tensor:
  # calculate attention mask for SW-MSA
  img_mask = torch.zeros((1,x_shape[1], x_shape[2], x_shape[3], 1))  # 1 Z H W 1 [1,8,186,360,1] (2,6,12)
  z_slices = (slice(0, -window_size[0]),
              slice(-window_size[0], -window_size[0]//2),
              slice(-window_size[0]//2, None))
  h_slices = (slice(0, -window_size[1]),
              slice(-window_size[1], -window_size[1]//2),
              slice(-window_size[1]//2, None))

  cnt = 0
  for z in z_slices:
    for h in h_slices:
        img_mask[:,z, h, :, :] = cnt
        cnt += 1
  mask_windows = window_partition(img_mask, window_size)  # nW,window_size, window_size, window_size, 1
  mask_windows=mask_windows.contiguous().view(list(mask_windows.shape[:4])+[-1]) #[1,4,31,30,144]
  attn_mask = mask_windows.unsqueeze(4) - mask_windows.unsqueeze(5) #[1,4,31,30,144,144]
  attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
  attn_mask=attn_mask.contiguous().permute(0,3,1,2,4,5).reshape(1,x_shape[3] // window_size[2],(x_shape[1]//window_size[0])*(x_shape[2] // window_size[1]),1,window_size[0] * window_size[1]*window_size[2],window_size[0] * window_size[1]*window_size[2])
  return attn_mask
 
def LoadConstantMask():
    land_mask=np.load("./constant_masks/land_mask.npy")
    soil_type=np.load("./constant_masks/soil_type.npy")
    topography=np.load("./constant_masks/topography.npy")
    land_mask=torch.from_numpy(land_mask).unsqueeze(0).float()
    soil_type=torch.from_numpy(soil_type).unsqueeze(0).float()
    topography=torch.from_numpy(topography).unsqueeze(0).float()
    tarns1 = transforms.Normalize(land_mask.mean(),land_mask.std())
    tarns2 = transforms.Normalize(soil_type.mean(),soil_type.std())
    tarns3 = transforms.Normalize(topography.mean(),topography.std())
    return tarns1(land_mask),tarns2(soil_type),tarns3(topography)

class Mlp(nn.Module):
  def __init__(self, dim, dropout_rate):
    super(Mlp,self).__init__()
    '''MLP layers, same as most vision transformer architectures.'''
    self.linear1 = nn.Linear(dim, dim * 4)
    self.linear2 = nn.Linear(dim * 4, dim)
    self.activation = nn.GELU()
    self.drop = nn.Dropout(dropout_rate)
    
  def forward(self, x):
    x = self.linear1(x)
    x = self.activation(x)
    x = self.drop(x)
    x = self.linear2(x)
    x = self.drop(x)
    return x
  
class PatchEmbedding(nn.Module):

  def __init__(self, dim):
    super(PatchEmbedding,self).__init__()
    '''Patch embedding operation'''
    # Zeor-pad
    self.pad=nn.ConstantPad3d((0,0,0,3,0,1),0)
    self.pad_surface=nn.ZeroPad2d((0,0,0,3))

    # Here we use convolution to partition data into cubes
    self.dim = dim
    self.conv = nn.Conv1d(in_channels=192, out_channels=dim, kernel_size=1, stride=1)
    self.conv_surface = nn.Conv1d(in_channels=112, out_channels=dim, kernel_size=1, stride=1)
    
    # Load constant masks from the disc
    land_mask, soil_type, topography = LoadConstantMask()
    self.forcing=nn.Parameter(torch.concatenate([land_mask.unsqueeze(0), soil_type.unsqueeze(0), topography.unsqueeze(0)],dim=1),requires_grad=False)
    self.input_concat=nn.Parameter(torch.randn(1,1,13,721,1440),requires_grad=True)

  def forward(self, input:torch.Tensor, input_surface:torch.Tensor):
    # Concat
    input = torch.concatenate([input,self.input_concat.repeat(input.shape[0],1,1,1,1)],dim=1) # B,6,13,721,1440

    # Zero-pad the input
    input=self.pad(input) # B, 6, 14, 724, 1440

    # linear
    input=input.reshape(-1,6,7,2,181,4,360,4) # B, 12, 7, 2, 181, 4, 360, 4
    input=input.permute(0,1,3,5,7,2,4,6) # B, 12, 2, 4, 4, 7, 181, 360
    input=input.reshape(input.shape[0], 192,-1) # B, 2, 6*2*4*4, 7*181*360
    input = self.conv(input)
    input=input.reshape(input.shape[0],192,7,181,360)

    # Add three constant fields to the surface fields and Zero-pad
    input_surface = torch.concatenate([input_surface, self.forcing.repeat(input_surface.shape[0],1,1,1)],dim=1) # B, 7, 721, 1440
    input_surface=self.pad_surface(input_surface) # B, 7, 724, 1440

    input_surface=input_surface.reshape(-1,7,181,4,360,4) # B, 14, 181, 4, 360, 4
    input_surface=input_surface.permute(0,1,3,5,2,4) # B, 14, 4, 4, 181, 360
    input_surface=input_surface.reshape(input.shape[0], 112,-1) # B, 2, 7*4*4, 181*360    

    input_surface = self.conv_surface(input_surface)
    input_surface=input_surface.reshape(-1,192,1,181,360)

    # Concatenate the input in the pressure level, i.e., in Z dimension
    x =  torch.concatenate([input_surface,input], dim=2) # [B, 192, 8, 181, 360]

    # Reshape x for calculation of linear projections
    return x.permute(0, 2, 3, 4, 1) # [B, 8, 181, 360, 192]
  

class PatchRecovery(nn.Module):
  def __init__(self, dim):
    super(PatchRecovery,self).__init__()
    '''Patch recovery operation'''
    # Hear we use two transposed convolutions to recover data
    self.conv = nn.Conv1d(in_channels=dim, out_channels=160, kernel_size=1, stride=1)
    self.conv_surface = nn.Conv1d(in_channels=dim, out_channels=64, kernel_size=1, stride=1)
    
  def forward(self, x:torch.Tensor):
    # The inverse operation of the patch embedding operation, patch_size = (2, 4, 4) as in the original paper
    # Reshape x back to three dimensions

    ori_shape=x.shape #[B, 8, 41, 41, 384]
    x=x.reshape(ori_shape[0],-1,ori_shape[-1])
    x = x.permute(0, 2, 1)
    x = x.reshape(x.shape[0], ori_shape[-1], ori_shape[1], ori_shape[2], ori_shape[3]) #[B, 384, 8, 41, 41]
    
    # Call the transposed convolution
    output = self.conv(x[:, :, 1:, :, :].reshape(x.shape[0],x.shape[1],-1)) # [2, 160, 11767]
    output = output.reshape(x.shape[0],5,2,4,4,7,ori_shape[2],ori_shape[3]) # [2, 5, 2, 4, 4, 7, 41, 41]
    output=output.permute(0,1,5,2,6,3,7,4) # [2, 5, 7, 2, 41, 4, 41, 4]
    output=output.reshape(-1,5,14,ori_shape[2]*4,ori_shape[3]*4) # [2, 5, 14, 164, 164]

    output_surface = self.conv_surface(x[:, :, 0, :, :].reshape(x.shape[0],x.shape[1],-1))
    output_surface=output_surface.reshape(-1,4,4,4,ori_shape[2],ori_shape[3]) # [2, 4, 4, 4, 41, 41]
    output_surface=output_surface.permute(0,1,4,2,5,3)  # [2, 4, 41, 4, 41, 4]
    output_surface=output_surface.reshape(-1,4,ori_shape[2]*4,ori_shape[3]*4)
    return output, output_surface  
  

class EarthAttention3D(nn.Module):
  def __init__(self, dim, heads, dropout_rate, window_size,input_shape):
    super(EarthAttention3D,self).__init__()
    '''
    3D window attention with the Earth-Specific bias, 
    see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
    '''
    # Initialize several operations
    self.linear1 = nn.Linear(dim, out_features=dim*3, bias=True)
    self.linear2 = nn.Linear(dim, dim)
    self.softmax = nn.Softmax(dim=-1)
    self.dropout = nn.Dropout(dropout_rate)

    # Store several attributes
    self.head_number = heads
    self.dim = dim

    self.scale = (dim//heads)**-0.5
    self.window_size = window_size
    self.input_shape = input_shape

    # input_shape is current shape of the self.forward function
    # You can run your code to record it, modify the code and rerun it
    # Record the number of different window types
    self.type_of_windows = (self.input_shape[0]//window_size[0])*(self.input_shape[1]//window_size[1]) # 28

    # Initialize the tensors using Truncated normal distribution
    self.earth_specific_bias = torch.randn((2 * window_size[2] - 1) * window_size[1] * window_size[1] * window_size[0] * window_size[0], self.type_of_windows, heads) # [3312, 28, 6]
    self.earth_specific_bias = nn.Parameter(self.earth_specific_bias,requires_grad=True)
    self.earth_specific_bias = nn.init.trunc_normal_(self.earth_specific_bias, std=0.02) 

    # For each type of window, we will construct a set of parameters according to the paper
    # self.earth_specific_bias = ConstructTensor(shape=((2 * window_size[2] - 1) * window_size[1] * window_size[1] * window_size[0] * window_size[0], self.type_of_windows, heads))
    # Construct position index to reuse self.earth_specific_bias
    self.position_index = self._construct_index() # [20736]
    self.mask=None
    
  def _construct_index(self):
    ''' This function construct the position index to reuse symmetrical parameters of the position bias'''
    # Index in the pressure level of query matrix
    coords_zi = RangeTensor(self.window_size[0])
    # Index in the pressure level of key matrix
    coords_zj = -RangeTensor(self.window_size[0])*self.window_size[0]

    # Index in the latitude of query matrix
    coords_hi = RangeTensor(self.window_size[1])
    # Index in the latitude of key matrix
    coords_hj = -RangeTensor(self.window_size[1])*self.window_size[1]

    # Index in the longitude of the key-value pair
    coords_w = RangeTensor(self.window_size[2])

    # Change the order of the index to calculate the index in total
    coords_1 =torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w]))
    coords_2 = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w]))
    coords_flatten_1 = torch.flatten(coords_1, start_dim=1) 
    coords_flatten_2 = torch.flatten(coords_2, start_dim=1)
    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    coords = coords.permute(1, 2, 0)

    # Shift the index for each dimension to start from 0
    coords[:, :, 2] += self.window_size[2] - 1
    coords[:, :, 1] *= 2 * self.window_size[2] - 1
    coords[:, :, 0] *= (2 * self.window_size[2] - 1)*self.window_size[1]*self.window_size[1]

    # Sum up the indexes in three dimensions
    position_index = torch.sum(coords, dim=-1)

    # Flatten the position index to facilitate further indexing
    return torch.flatten(position_index)
    
  def forward(self, x: torch.Tensor, mask: torch.Tensor):
    # Record the original shape of the input
    original_shape = x.shape # [B, 4, 28, 144, 192]

    # Linear layer to create query, key and value
    # x=x.reshape(self.type_of_windows,self.window_size[0]*self.window_size[1]*self.window_size[2],self.dim self.head_number)
    x = self.linear1(x) # [2, 4, 28, 144, 576]

    # reshape the data to calculate multi-head attention
    qkv = x.view((x.shape[0], x.shape[1],x.shape[2],x.shape[3], 3, self.head_number, self.dim // self.head_number)) # [2, 4, 28, 144, 3, 6, 32]
    query, key, value = qkv.permute(4,0,1,2,5,3,6) # 3*[2, 4, 28, 6, 144, 32]
    # Scale the attention
    query = query * self.scale
    
    # Calculated the attention, a learnable bias is added to fix the nonuniformity of the grid.
    attention = query @ key.transpose(-2,-1) # @ denotes matrix multiplication [2, 4, 28, 6, 144, 144]

    # Add the Earth-Specific bias to the attention matrix
    EarthSpecificBias = self.earth_specific_bias[self.position_index]
    EarthSpecificBias = EarthSpecificBias.reshape(self.window_size[0]*self.window_size[1]*self.window_size[2], self.window_size[0]*self.window_size[1]*self.window_size[2], self.type_of_windows, self.head_number)
    EarthSpecificBias = EarthSpecificBias.permute(2, 3, 0, 1) # [28, 6, 144, 144]
    attention = attention + EarthSpecificBias
    
    # Mask the attention between non-adjacent pixels, e.g., simply add -100 to the masked element.
    attention = attention+mask
    attention = self.softmax(attention)
    attention = self.dropout(attention)
    
    # Calculated the tensor after spatial mixing.
    x = attention @ value # @ denote matrix multiplication [B, 4, 28, 6, 144, 32]

    # Reshape tensor to the original shape
    x = x.permute(0,1,2,4,3,5) # [2, 4, 7, 144, 6, 32]
    x = x.reshape(original_shape)

    # Linear layer to post-process operated tensor
    x = self.linear2(x)
    x = self.dropout(x)
    return x

class EarthSpecificBlock(nn.Module):
  def __init__(self, dim, drop_path_ratio, heads,x_shape):
    super(EarthSpecificBlock,self).__init__()
    '''
    3D transformer block with Earth-Specific bias and window attention, 
    see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
    The major difference is that we expand the dimensions to 3 and replace the relative position bias with Earth-Specific bias.
    '''
    # Define the window size of the neural network 
    self.window_size = (2, 6, 12)

    # Initialize serveral operations
    self.drop_path = timm.layers.DropPath(drop_path_ratio)
    self.norm1 = nn.LayerNorm(dim)
    self.norm2 = nn.LayerNorm(dim)
    self.linear = Mlp(dim, 0)

    x_shape=list(x_shape)
    self.pad_num_H=self.window_size[1]-(x_shape[2]%self.window_size[1]) if (x_shape[2]%self.window_size[1])!=0 else 0
    x_shape[2]=x_shape[2]+self.pad_num_H
    self.pad_num_W=self.window_size[2]-(x_shape[3]%self.window_size[2]) if (x_shape[3]%self.window_size[2])!=0 else 0
    x_shape[3]=x_shape[3]+self.pad_num_W

    self.attention = EarthAttention3D(dim, heads, 0, self.window_size,x_shape[1:])
    self.pad=nn.ZeroPad2d((0,self.pad_num_W,0,self.pad_num_H))
    self.mask=nn.Parameter(gen_mask(x_shape,self.window_size),False)

  def forward(self, x:torch.Tensor, roll):
    # Save the shortcut for skip-connection
    shortcut = x

    # Zero-pad input if needed
    x=x.permute(0,1,4,2,3) # [B, 8, 192, 41, 41]
    x = self.pad(x)
    x=x.permute(0,1,3,4,2) # [B, 8, 42, 48, 192]

    # Store the shape of the input for restoration
    ori_shape = x.shape
    
    if roll:
      # Roll x for half of the window for 3 dimensions
      x = torch.roll(x,shifts=[-self.window_size[0]//2,-self.window_size[1]//2,-self.window_size[2]//2],dims=[1,2,3])
      # Generate mask of attention masks
      # If two pixels are not adjacent, then mask the attention between them
      # Your can set the matrix element to -1000 when it is not adjacent, then add it to the attention

    # Reorganize data to calculate window attention [B, 4, 2, 7, 6, 4, 12, 192]
    x_window = x.reshape(x.shape[0], x.shape[1]//self.window_size[0], self.window_size[0], x.shape[2] // self.window_size[1], self.window_size[1], x.shape[3] //self. window_size[2], self.window_size[2], x.shape[-1])
    x_window = x_window.permute(0,5,1,3,2,4,6,7) # [B, 4, 4, 7, 2, 6, 12, 192]
    tmp_shape=x_window.shape
    # Get data stacked in 3D cubes, which will further be used to calculated attention among each cube [B, NW, NZ*NH, Windowsize, C]
    x_window = x_window.reshape(x_window.shape[0],x_window.shape[1],x_window.shape[2]*x_window.shape[3],x_window.shape[4]*x_window.shape[5]*x_window.shape[6],x_window.shape[7]) # [B, 4, 28, 144, 192]

    # Apply 3D window attention with Earth-Specific bias
    x_window = self.attention(x_window, self.mask if roll else 0)
    # Reorganize data to original shapes
    x = x_window.reshape(tmp_shape) # [1,30,4,31,2,6,12,192]
    x = x.permute(0, 2, 4, 3, 5, 1, 6, 7) # [1, 4, 2, 31, 6, 30, 12, 192]
    # Reshape the tensor back to its original shape
    x = x.reshape(ori_shape)
    if roll:
      # Roll x back for half of the window
      x = torch.roll(x,shifts=[self.window_size[0]//2, self.window_size[1]//2, self.window_size[2]//2],dims=[1,2,3])

    # Crop the zero-padding
    x = x[:,:,:x.shape[2]-self.pad_num_H,:x.shape[3]-self.pad_num_W]
    # Reshape the tensor back to the input shape
    # Main calculation stages
    x = shortcut + self.drop_path(self.norm1(x))

    x = x + self.drop_path(self.norm2(self.linear(x)))
    return x
  
class EarthSpecificLayer(nn.Module):
  def __init__(self, depth, dim, drop_path_ratio_list, heads, x_shape):
    super(EarthSpecificLayer,self).__init__()
    '''Basic layer of our network, contains 2 or 6 blocks'''
    self.depth = depth
    self.blocks = nn.ModuleList()

    # Construct basic blocks
    for i in range(depth):
      self.blocks.append(EarthSpecificBlock(dim, drop_path_ratio_list[i], heads, x_shape))
      
  def forward(self, x):
    for i in range(self.depth):
      # Roll the input every two blocks
      if i % 2 == 0:
        x=self.blocks[i](x, roll=False)
      else:
        x=self.blocks[i](x, roll=True)
    return x
  
class DownSample(nn.Module):
  def __init__(self, dim):
    super(DownSample,self).__init__()
    '''Down-sampling operation'''
    # A linear function and a layer normalization
    self.pad=nn.ZeroPad2d((0,0,0,1))
    self.linear = nn.Linear(4*dim, 2*dim, bias=False)
    self.norm = nn.LayerNorm(4*dim)

  def forward(self, x:torch.Tensor):
    # Padding the input to facilitate downsampling
    x=x.permute(0,4,1,2,3) # [B, 192, 8, 41, 41]
    x = self.pad(x)
    x=x.permute(0,2,3,4,1) # [B, 8, 42, 42, 192]

    # Reorganize x to reduce the resolution: simply change the order and downsample
    _,Z, H, W,_ = x.shape
    # Reshape x to facilitate downsampling
    x = x.reshape(x.shape[0], Z, H//2, 2, W//2, 2, x.shape[-1]) # [B, 8, 21, 2, 21, 2, 192]
    # Change the order of x
    x = x.permute(0,1,2,4,3,5,6) # [B, 8, 21, 21, 2, 2, 192]

    x = x.reshape(x.shape[0], Z,(H//2),(W//2), 4 * x.shape[-1])
    # Call the layer normalization
    x = self.norm(x)
    # Decrease the channels of the data to reduce computation cost
    x = self.linear(x) # [2, 8, 21, 21, 384]
    return x
  
class UpSample(nn.Module):
  def __init__(self, input_dim, output_dim,crop_size=(181,None)):
    super(UpSample,self).__init__()
    self.crop_size=crop_size
    '''Up-sampling operation'''
    # Linear layers without bias to increase channels of the data
    self.linear1 = nn.Linear(input_dim, output_dim*4, bias=False)
    # Linear layers without bias to mix the data up
    self.linear2 = nn.Linear(output_dim, output_dim, bias=False)
    # Normalization
    self.norm = nn.LayerNorm(output_dim)
  
  def forward(self, x:torch.Tensor):
    # Call the linear functions to increase channels of the data
    x = self.linear1(x) # [B, 8, 21, 21, 768]
    # Reorganize x to increase the resolution: simply change the order and upsample from (8, 91, 180) to (8, 182, 360)
    # Reshape x to facilitate upsampling.
    x = x.reshape(x.shape[0], x.shape[1], 91, 180, 2, 2, x.shape[-1]//4)
    # Change the order of x
    x = x.permute(0,1,2,4,3,5,6)
    # Reshape to get Tensor with a resolution of (8, 182, 360)
    x = x.reshape(x.shape[0], x.shape[1], 182, 360, x.shape[-1])
    # Crop the output to the input shape of the network
    x = x[:,:,:181,:360]
    # Reshape x back
    # Call the layer normalization
    x = self.norm(x)
    # Mixup normalized tensors
    x = self.linear2(x) # [2, 8, 41, 41, 192]
    return x
  
class ForeCastModel(nn.Module):

  def __init__(self,
               dim=192,
               depth=[2,6,6,2],
               head_number=[6,12,12,6]):
    super(ForeCastModel,self).__init__()

    # Drop path rate is linearly increased as the depth increases
    drop_path_list =np.linspace(0, 0.2, 8)

    # Patch embedding
    self._input_layer = PatchEmbedding(dim)

    # Four basic layers
    self.layer1 = EarthSpecificLayer(depth[0], dim,   drop_path_list[:2], head_number[0],  [1, 8, 181, 360, dim])
    self.layer2 = EarthSpecificLayer(depth[1], dim*2, drop_path_list[(8-depth[1]):], head_number[1],  [1, 8, 91, 180, dim*2])
    self.layer3 = EarthSpecificLayer(depth[2], dim*2, drop_path_list[(8-depth[2]):], head_number[2],  [1, 8, 91, 180, dim*2])
    self.layer4 = EarthSpecificLayer(depth[3], dim,   drop_path_list[:2], head_number[3],  [1, 8, 181, 360, dim])

    # Upsample and downsample
    self.upsample = UpSample(dim*2, dim)
    self.downsample = DownSample(dim)

    # Patch Recovery
    self._output_layer = PatchRecovery(dim*2)

  def forward(self, input, input_surface):
  # def forward(self, x):
    '''Backbone architecture'''
    # Embed the input fields into patches
    x = self._input_layer(input, input_surface) # [B, 8, 41, 41, 192]
    # Encoder, composed of two layers
    # Layer 1, shape (8, 360, 181, C), C = 192 as in the original paper
    x = self.layer1(x) 
    
    # Store the tensor for skip-connection
    skip = x
    # Downsample from (8, 181, 360) to (8, 91, 180)
    x = self.downsample(x)
    # Layer 2, shape (8, 91, 180, 2C), C = 192 as in the original paper
    x = self.layer2(x) 
    # Decoder, composed of two layers
    # Layer 3, shape (8, 91, 180, 2C), C = 192 as in the original paper
    x = self.layer3(x) 
    # Upsample from (8, 91, 180) to (8, 181, 360)
    x = self.upsample(x)
    # Layer 4, shape (8, 181, 360, 2C), C = 192 as in the original paper
    x = self.layer4(x) 
    # Skip connect, in last dimension(C from 192 to 384)

    x = torch.concatenate([skip, x],dim=-1) # [2, 8, 41, 41, 384]
    # Recover the output fields from patches
    output, output_surface  = self._output_layer(x)
    # Crop the output to remove zero-paddings
    return output[:,:,:13,:721,:1440], output_surface[:,:,:721,:1440]
  
  def load_weight(self,path:str):
    state_dict=torch.load(path,map_location="cpu")
    new_state_dict={}
    for k,v in state_dict.items():
      if k.endswith("_input_layer.forcing"):
        new_state_dict[k]=v[:,:,:721,:1440]
      elif k.endswith("_input_layer.input_concat"):
        new_state_dict[k]=v[0,:,:,:,:721,:1440].clone()
      elif "_input_layer.input_concat2" in k:
        continue
      else:
        new_state_dict[k]=v
    self.load_state_dict(new_state_dict,False) 


if __name__ =='__main__':
    module=PanguModel().cuda().eval()
    # module.load_weight("/data/kaixin/Pangu-Weather/pretrained/pangu_weather_6.pth") 
    
    mean=np.load("./mean.npy").transpose(3,0,1,2).astype(np.float32)[:,::-1]
    std=np.load("./std.npy").transpose(3,0,1,2).astype(np.float32)[:,::-1]
    mean_surface=np.array([ 1.0095783e+05,-5.0557569e-02, 1.8899263e-01, 2.7844345e+02]).reshape(4,1,1).astype(np.float32)
    std_surface=(np.array([1320.9279,5.4929,4.712085,21.501575]).reshape(4,1,1)).astype(np.float32)

    input = np.load('./input_upper.npy').astype(np.float32)
    input_surface = np.load("./input_surface.npy").astype(np.float32)

    input= (input-mean)/std
    input_surface= (input_surface-mean_surface)/std_surface

    input = torch.from_numpy(input).unsqueeze(0).cuda()
    input_surface = torch.from_numpy(input_surface).unsqueeze(0).cuda()

    with torch.no_grad():
        out,out_surface=module(input,input_surface)  

    out=out.detach().cpu().numpy()*std+mean
    out_surface=out_surface.detach().cpu().numpy()*std_surface+mean_surface   

    np.save("./output_upper_torch.npy",out)
    np.save("./output_surface_torch.npy",out_surface)

    import pdb;pdb.set_trace()