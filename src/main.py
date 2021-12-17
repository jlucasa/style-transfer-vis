'''
Style Transfer Neural Net Visualization Project
CS 6965: Advanced Data Visualization
University of Utah: Dr. Bei Wang-Phillips

Alan Weber, Jared Amen, Pranav Rajan
Fall 2021

This script runs a Streamlit server on localhost with a user interface
that allows a user to input a content image and style image (along with
weights, emphasized layers, and other applicable hyperparameters)
for the purposes of style transfer (as in 
http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).

The purpose of this playground is to implement intermediate activation maps
and resultant images for the user as the neural net is being ran, exposing
details of the "black box" that is typically deep learning. By default, the
first 16 channels of all 13 layers are displayed using a spectral colormap,
but the user has the option of displaying these channels using a spectral,
grayscale, rainbow, and jet colormap (as defined by matplotlib).
'''

# Libraries
import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import PIL
import numpy as np
import matplotlib.pyplot as plt
import re
import zipfile
import os
import glob
import time
import datetime

# Supplementary constants for image/tensor conversion
SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406])
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225])
DTYPE = torch.cuda.FloatTensor

# Pretrained CNN for feature extraction
CNN = torchvision.models.squeezenet1_1(pretrained=True).features
CNN.type(DTYPE)

# Default weights for UI
WEIGHTS_MAP = {
    1: 300000,
    4: 1000,
    6: 15,
    7: 3
}

CHANNEL_BREAK_MAP = {
    'first_16': {
        'func': lambda i, num_channels: i == 16,
        'should_continue': False,
        'num_rows': 2,
        'num_cols': lambda num_channels: 8
    },
    'last_16': {
        'func': lambda i, num_channels: i < num_channels - 16,
        'should_continue': True,
        'num_rows': 2,
        'num_cols': lambda num_channels: 8
    },
    'first_32': {
        'func': lambda i, num_channels: i == 32,
        'should_continue': False,
        'num_rows': 4,
        'num_cols': lambda num_channels: 8
    },
    'last_32': {
        'func': lambda i, num_channels: i < num_channels - 32,
        'should_continue': True,
        'num_rows': 4,
        'num_cols': lambda num_channels: 8
    },
    # 'every_other': {
    #     'func': lambda i, num_channels: i % 2 == 0,
    #     'should_continue': True,
    #     'num_rows': lambda num_channels: max(num_channels // 8, 1)
    # },
    # 'every_first': {
    #     'func': lambda i, num_channels: i % 2 == 1,
    #     'should_continue': True,
    #     'num_rows': lambda num_channels: max(num_channels // 8, 1)
    # },
    'every_eighth': {
        'func': lambda i, num_channels: i % 8 != 0,
        'should_continue': True,
        'num_rows': -64,
        'num_cols': lambda num_channels: 8
    },
    'every_sixteenth': {
        'func': lambda i, num_channels: i % 16 != 0,
        'should_continue': True,
        'num_rows': -128,
        'num_cols': lambda num_channels: 8 if num_channels > 64 else 4
    }
}

def preprocess_image(img, size=512):
    '''
    Preprocesses the input image for the style transfer.

    Inputs:
    - img: PIL image of shape (1, 3, H, W)
    - size: (1, 3, H, W) with H, W = imgsize

    Returns:
    - Tensor of shape (1, 3, H, W)
    '''

    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])

    return transform(img)


def deprocess_image(img):
    '''
    De-process an image so that it looks good in the original scale.

    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding a batch of images

    Returns:
    - deprocessed_img: a PyTorch Tensor of shape (3, H, W) holding the deprocessed image
    '''

    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in SQUEEZENET_STD.tolist()]),
        T.Normalize(mean=[-m for m in SQUEEZENET_MEAN.tolist()], std=[1, 1, 1]),
        T.Lambda(rescale),
        T.ToPILImage(),
    ])

    return transform(img)


def rescale(x):
    '''
    Rescale an image tensor of shape (N, C, H, W) to be within the range [0, 1]

    Inputs:
    - x: PyTorch Tensor of shape (N, C, H, W) holding images to be rescaled.

    Returns:
    - Tensor of shape (N, C, H, W)
    '''

    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


def features_from_img(imgpath, imgsize):
    '''
    Read an image from imgpath, resize it to imgsize, and extract features.

    Inputs:
    - imgpath: path to the image file
    - imgsize: size to resize the image to: (imgsize, imgsize)

    Returns:
    - img_var: a PyTorch Variable of shape (1, C, H, W)
    - features: a list of PyTorch Variables, representing the features
        from the CNN; each has shape (1, C_i, H_i, W_i)
    '''

    img = preprocess_image(PIL.Image.open(imgpath), size=imgsize)
    img_var = img.type(DTYPE)
    return extract_features(img_var), img_var
    

def extract_features(x, cnn=CNN):
    '''
    Use the CNN to extract features from the input image x.
    
    Inputs:
    - x: A PyTorch Tensor of shape (N, C, H, W) holding a minibatch of images that
      will be fed to the CNN.
    - cnn: A PyTorch model that we will use to extract features.
    
    Returns:
    - features: A list of feature for the input images x extracted using the cnn model.
      features[i] is a PyTorch Tensor of shape (N, C_i, H_i, W_i); recall that features
      from different layers of the network may have different numbers of channels (C_i) and
      spatial dimensions (H_i, W_i).
    '''
    features = []
    prev_feat = x

    for i, module in enumerate(cnn._modules.values()):
        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    
    return features


def content_loss(content_weight, content_current, content_original):
    '''
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_target: features of the content image, Tensor with shape (1, C_l, H_l, W_l).
    
    Returns:
    - scalar content loss
    '''
    shape = content_current.shape
    print('content shape: ', shape)
    dim = shape[1] * shape[2] * shape[3]

    F = content_current.reshape(dim)
    P = content_original.reshape(dim)

    lc = sum(((F - P) ** 2))
    lc *= content_weight

    return lc


def gram_matrix(features, normalize=True):
    '''
    Compute the Gram matrix from features.
    
    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    '''
    N, C, H, W = features.shape
    M = H * W
    G = torch.zeros(N, C, C)
    print('In Gram matrix: N C H W: ', N, C, H, W)

    for n in range(N):
        F = features[n].reshape(C, M)
        G[n] = torch.mm(F, F.t())
    
    if normalize:
        G /= (C * H * W)
    
    return G


def style_loss(feats, style_layers, style_targets, style_weights):
    '''
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    '''
    n_layers = len(style_layers)
    print('In style loss - # of style layers: ', n_layers)
    l_tot = 0

    for i in range(n_layers):
        layer = style_layers[i]
        GL = gram_matrix(feats[layer]).flatten()
        AL = style_targets[i].flatten()
        L = style_weights[i] * sum(((GL - AL) ** 2))
        l_tot += L

    return l_tot


def tv_loss(img, tv_weight):
    '''
    Compute total variation loss.
    
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    '''
    image = img
    image1 = image[:, :, 1:, :].flatten()
    image2 = image[:, :, :-1, :].flatten()
    image3 = image[:, :, :, 1:].flatten()
    image4 = image[:, :, :, :-1].flatten()

    return tv_weight * (sum((image2 - image1) ** 2) + sum((image4 - image3) ** 2))


# def change_color_mapping(fig, cmap):
#     for ax in fig.axes:
#         ax.set_cmap(cmap)


def download_all_figs_button(figs, epoch, activation_container):
    '''
    For a set of download activation maps, creates a zip file in the root
    directory of the server and a download button for that zip file

    Args:
    - figs: A list of MPL figures to write to the zip
    - epoch: The current testing epoch of the model
    - activation_container: The output container for the download button
    '''
    zip_file_name = f'figs_for_epoch_{epoch}.zip'
    with zipfile.ZipFile(zip_file_name, 'w') as myzip:
        for i, fig in enumerate(figs):
            fig.savefig(f'activation_maps_epoch{epoch}_layer{i}.png')
            myzip.write(f'activation_maps_epoch{epoch}_layer{i}.png')
            os.remove(f'activation_maps_epoch{epoch}_layer{i}.png')

    with open(zip_file_name, 'rb') as zip_file:
        activation_container.download_button(
            'Download activation maps', 
            zip_file,
            file_name=zip_file_name, 
            mime='application/zip'
        )


def remove_all_previous_figs():
    '''
    Removes all previous figures in the top-level directory of the server. Meant for basic
    cleaning of state upon rerunning the playground.
    '''

    def get_files():
        fig_filepaths = []
        paths = glob.glob('figs_for_epoch_*.zip')
        fig_filepaths += paths

        return fig_filepaths
    
    fig_filepaths = get_files()

    for filepath in fig_filepaths:
        os.remove(filepath)


def layer_vis(
    feats, 
    num_epoch, 
    output_container, 
    feats_choices, 
    channel_break_condition, 
    should_continue_on_channels, 
    num_rows,
    num_cols,
    color_mapping
):
    '''
    Visualize intermediate layers using Matplotlib. By default,
    visualizes the first 16 channels of each layer.

    Args:
    - feats: A list of features from extract_features().
    - num_epoch: The epoch number.
    - output_container: The Streamlit container to output activation maps to.
    - channel_break_condition: The lambda function at which output for the layer should either break or continue (see CHANNEL_BREAK_MAP)
    - should_continue_on_channels: Whether the output for the layer should break or continue (see CHANNEL_BREAK_MAP)
    - num_rows: Hard-coded number of rows for the MPL figure, based on the channel vis selection (see CHANNEL_BREAK_MAP)
    - num_cols: Lambda function for number of cols of the MPL figure, based on channel vis selection (see CHANNEL_BREAK_MAP)
    - color_mapping: A list of color mappings.
    '''
    activation_container = output_container.expander(f'Activation Maps for Epoch {num_epoch}')
    figs = []

    for num_layer in feats_choices:
        num_channels = len(feats[num_layer][0, :])
        fig, axes = plt.subplots(num_rows if num_rows > 0 else max(num_channels // -num_rows, 1), num_cols(num_channels), figsize=(50, 10), squeeze=False)  

        fig.suptitle(f'Activation Maps for Layer {num_layer}', fontsize=36)

        layer_vis = feats[num_layer][0, :, :, :].data.cpu()
        row = 0
        col = 0
        
        for i, filter in enumerate(layer_vis):
            if channel_break_condition(i, num_channels):
                if should_continue_on_channels:
                    continue
                else:
                    break

            if col % num_cols(num_channels) == 0 and col != 0:
                row += 1

            axes[row, col % num_cols(num_channels)].axis('off')
            axes[row, col % num_cols(num_channels)].imshow(filter, cmap=color_mapping)
            axes[row, col % num_cols(num_channels)].title.set_text(f'Channel {i}')

            col += 1

        activation_container.pyplot(fig)
        figs.append(fig)
        plt.close()

    # download_all_figs_button(figs, num_epoch, activation_container)


def style_transfer(
    content_img: str or st.UploadedFile, 
    style_img: str or st.UploadedFile, 
    content_size=192, 
    style_size=192,
    style_layers=[1, 4, 6, 7], 
    content_layer=3,
    content_weight=0.06, 
    style_weights=[300000, 1000, 15, 3],
    tv_weight=0.02,
    initial_lr=3.0,
    decayed_lr=0.10,
    decay_lr_at=9,
    layer_vis_choices=[i for i in range(len(CNN))],
    channel_vis_choice='first_16',
    num_epochs=10,
    init_random=False,
    observe_intermediate_result_count=2,
    color_mapping='nipy_spectral'
):
    '''
    Run style transfer using pretrained SqueezeNet!
    
    Inputs:
    - content_image: filename of content image
    - style_image: filename of style image
    - image_size: size of smallest image dimension (used for content loss and generated image)
    - style_size: size of smallest style image dimension
    - content_layer: layer to use for content loss
    - content_weight: weighting on content loss
    - style_layers: list of layers to use for style loss
    - style_weights: list of weights to use for each layer in style_layers
    - tv_weight: weight of total variation regularization term
    - init_random: initialize the starting image to uniform random noise
    - num_epochs: number of epochs to run
    - observe_intermediate_result_count: number of epochs to observe intermediate results
    - color_mapping: color mapping to use for generated image
    - layer_vis_choices: list of layer choices to display activation maps for
    - channel_vis_choice: channel criterion to use for activation map display (see CHANNEL_BREAK_MAP) 
    '''

    output_container = st.container()
    exception_message = output_container.empty()

    try:
        start_time = time.time()

        content_img_as_tensor = preprocess_image(PIL.Image.open(content_img), size=content_size).type(DTYPE)
        feats = extract_features(content_img_as_tensor)
        content_target = feats[content_layer].clone()

        style_img_as_tensor = preprocess_image(PIL.Image.open(style_img), size=style_size).type(DTYPE)
        feats = extract_features(style_img_as_tensor)
        style_targets = []

        for idx in style_layers:
            style_targets.append(gram_matrix(feats[idx].clone()))
        
        # Initialize output image to content image or noise
        if init_random:
            img = torch.Tensor(content_img_as_tensor.size()).uniform_(0, 1).type(DTYPE)
        else:
            img = content_img_as_tensor.clone().type(DTYPE)
        
        # We do want the gradient computed on our image!
        img.requires_grad_()

        # # Set up optimization hyperparameters
        # initial_lr = 3.0
        # decayed_lr = 0.1
        # decay_lr_at = 180

        # Note that we are optimizing the pixel values of the image by passing
        # in the img Torch tensor, whose requires_grad flag is set to True
        optimizer = torch.optim.Adam([img], lr=initial_lr)

        output_container.markdown('## Image Outputs')
        progress_bar = output_container.progress(0.0)
        info_message = output_container.info('Starting style transfer...')
        output_container.markdown('### Composite Images')
        img_container = output_container.expander('Outputs')
        activation_map_header = output_container.empty()
        display_activation_map_header = True
        insert_img_in_col1 = True

        print(f'num epochs: {num_epochs}')

        for t in range(num_epochs):
            progress_bar.progress((t + 1) / num_epochs)

            print('epoch: ', t)
            epoch_start_time = time.time()

            if t < 190:
                img.data.clamp_(-1.5, 1.5)
            optimizer.zero_grad()

            # check_time = time.time()

            feats = extract_features(img)

            # check_time = time.time()

            c_loss = content_loss(content_weight, feats[content_layer], content_target)
            s_loss = style_loss(feats, style_layers, style_targets, style_weights)
            t_loss = tv_loss(img, tv_weight)
            loss = c_loss + s_loss + t_loss

            loss.backward()

            if t == decay_lr_at:
                optimizer = torch.optim.Adam([img], lr=decayed_lr)
            
            optimizer.step()

            if t % observe_intermediate_result_count == 0:
                info_message.info(f'Plotting activation maps for epoch {t}...')
                col1, col2 = img_container.columns(2)

                if insert_img_in_col1:
                    with col1:
                        img_container.image(deprocess_image(img.data.cpu()), caption=f'Image at Epoch {t}')
                        insert_img_in_col1 = False
                else:
                    with col2:
                        img_container.image(deprocess_image(img.data.cpu()), caption=f'Image at Epoch {t}')
                        insert_img_in_col1 = True
                
                if display_activation_map_header:
                    activation_map_header.markdown('### Activation Maps')
                    display_activation_map_header = False
                    
                layer_vis(
                    feats, 
                    t, 
                    output_container,
                    layer_vis_choices, 
                    CHANNEL_BREAK_MAP[channel_vis_choice]['func'], 
                    CHANNEL_BREAK_MAP[channel_vis_choice]['should_continue'],
                    CHANNEL_BREAK_MAP[channel_vis_choice]['num_rows'],
                    CHANNEL_BREAK_MAP[channel_vis_choice]['num_cols'],
                    color_mapping
                )
                # st.image(deprocess_image(img.data.cpu()), caption=f'Image at Epoch {t + 1}')

            info_message.info(f'Epoch {t} completed. Total elapsed time: {datetime.timedelta(seconds=round(time.time() - epoch_start_time, 2))}')

        output_container.markdown('## Final Image')
        output_container.image(deprocess_image(img.data.cpu()), caption='Final Image')
        
        st.balloons()
        info_message.success(f'Finished! Total elapsed time: {datetime.timedelta(seconds=round(time.time() - start_time, 2))}')
    except Exception as e:
        exception_message.exception(e)


def run_style_transfer(**args):
    '''
    Run style transfer using pretrained SqueezeNet (with error checking)!

    This function should only be called from the UI

    Inputs:
    - content_img: filename of content image
    - style_img: filename of style image
    - image_size: size of smallest image dimension (used for content loss and generated image)
    - style_size: size of smallest style image dimension
    - content_layer: layer to use for content loss
    - content_weight: weighting on content loss
    - style_layers: list of layers to use for style loss
    - style_weights: list of weights to use for each layer in style_layers
    - tv_weight: weight of total variation regularization term
    - init_random: initialize the starting image to uniform random noise
    - num_epochs: number of epochs to run
    - observe_intermediate_result_count: number of epochs to observe intermediate results
    - color_mapping: color mapping to use for generated image
    - layer_vis_choices: list of layer choices to display activation maps for
    - channel_vis_choice: channel criterion to use for activation map display (see CHANNEL_BREAK_MAP)
    '''

    if args['content_img'] is None:
        args['content_img'] = 'assets/input-imgs/tubingen.jpg'
    
    if args['style_img'] is None:
        args['style_img'] = 'assets/input-imgs/starry_night.jpg'

    for key in args:
        if args[key] is None: 
            st.error(f'Missing fields: {[key for key in args if args[key] is None]}')
            st.error('Please fill in all fields')
            
            return

    style_transfer(**args)


def main():
    """
    Main function
    """

    # Available layers and channel visualization methods
    available_layers = [i for i in range(len(CNN))]
    available_vis_channel_methods = [key for key in CHANNEL_BREAK_MAP.keys()]

    # Page Config
    st.set_page_config(page_title='Style Transfer Vis', page_icon=':art:')

    # Containers
    initial_imgs_container = st.container()

    # Markdown Description of Project
    st.title('Style Transfer Neural Network Visualization')
    st.markdown('## Jared Amen, Pranav Rajan, and Alan Weber')
    st.markdown('## CS 6965: Advanced Data Visualization -- Fall 2021')
    st.markdown('### Professor Bei Wang-Phillips -- University of Utah')

    st.markdown('## How to Use This Workbench')
    st.markdown('''
        **Note:** If you are using this software with our default images (`tubingen.jpg` and `starry_night.jpg` -- see below)), you can simply 
        run style transfer with the default parameters, as the UI is set up by default for optimal handling of hyperparameters and these images.
    ''')

    st.markdown('### Step 1: Upload a Content and Style Image')
    st.markdown('''
        You can upload any content and style image (**in JPG format**) you\'d like. Keep in mind that the content image is the "destination" image 
        and the style image is the "source" image.
    ''')

    st.markdown('### Step 2: Choose Image Sizes')
    st.markdown('''
        The image size is the smallest dimension of the image. For example, if you want to use the image `tubingen.jpg` as the content image,
        you can choose the image size to be the smallest dimension of the image (e.g. `192`). This will ensure that the content loss is minimized. 
        If you choose a larger image size, the style transfer will take longer to run, but the resulting image will be more detailed.
    ''')

    st.markdown('### Step 3: Select Style Layers and Weights')
    st.markdown('''
        Specify a list of which layers to use for style loss. Each layer corresponds to the respective feature as in the summary [here](#squeezenet-feature-summary).
        Each layer is assigned an inputted weight, which is used to calculate the style loss for the respective layer. We generally use higher weights for 
        the earlier style layers because they describe more local/smaller scale features, which are more important to texture than features over larger 
        receptive fields. In general, increasing these weights will make the resulting image look less like the original content and more distorted towards 
        the appearance of the style image.
    ''')

    st.markdown('### Step 4: Select Content Layer')
    st.markdown('''
        Specify a layer you want to use for content loss. Each layer corresponds to the respective feature as in the summary [here](#squeezenet-feature-summary).
        The layer is assigned an inputted weight, which is used to calculate the content loss for the respective layer. Increasing the value of this parameter 
        will make the final image look more realistic (closer to the original content).
    ''')

    st.markdown('### Step 5: Select Number of Epochs')
    st.markdown('Select the number of epochs you want to run style transfer for. Higher epoch counts will take longer to run, but the resulting image will be more detailed.')

    st.markdown('### Step 6: Select Total Variation Weight')
    st.markdown('''
        Specify the total variation regularization weight in the overall loss function. Increasing this value makes the resulting image look smoother and less jagged, 
        at the cost of lower fidelity to style and content.
    ''')

    st.markdown('### Step 7: Select Learning Rate Hyperparameters')
    st.markdown('''
        Specify learning rate hyperparameters for the optimizer. They are defined as such:
        - `initial_lr`: The learning rate for the optimizer
        - `decay_lr_at`: The epoch # at which to decay the learning rate to `decayed_lr` (by default, this is 90 percent of the total epoch count specified)
        - `decayed_lr`: The decayed learning rate for the optimizer (after `decay_lr_at` epochs)
    ''')

    st.markdown('### Step 8: Modify Output Frequency and Structure of Intermediate Results')
    st.markdown('''
        Specify details related to output frequency and structure of intermediate results. This workbench supports the following specifications:
        - *Epoch Frequency for Observing Intermediate Results.* This defines how often the workbench will output intermediate resultant images and activation maps.
        - *Intermediate Layers to Visualize.* This defines which layers to visualize for each intermediate result.
        - *Channel Criterion for Activation Maps.* This defines which channels to visualize for each layer. This workbench supports the following specifications for channel criteria:
            - `first_16`: Visualize the first 16 channels of the layer
            - `last_16`: Visualize the last 16 channels of the layer
            - `first_32`: Visualize the first 32 channels of the layer
            - `last_32`: Visualize the last 32 channels of the layer
            - `every_eighth`: Visualize every 8th channel of the layer
            - `every_sixteenth`: Visualize every 16th channel of the layer
        - *Color Mapping.* This defines how to color the activation maps. This workbench supports the following specifications for color mappings:
            - `nipy_spectral`: Spectral Color Mapping
            - `jet`: Jet Color Mapping
            - `gray`: Grayscale Color Mapping
            - `rainbow`: Rainbow Color Mapping
        - *Random Initialization.* This defines whether to use random noise or the content image as an initial image.
    ''')

    st.markdown('### Step 9: Run Style Transfer')
    st.markdown('''
        From here, you can run style transfer! The output as you have specified it will be written to the content display of this workbench.
        You can download individual outputs by right-clicking on the image and saving it.
    ''')

    st.markdown('## Project Introduction')
    
    st.markdown('### Background and Motivation')
    st.markdown('''
        - Artificial Neural Networks are increasingly prevalent in analytic solutions for many industries
        - Understanding their “decision making process” is a key factor in the adoption of this technology
        - Visual interpretation of their operation constitutes a big step in this direction
        - The “style transfer” use case for convolutional neural nets (CNNs) is visually compelling and germane for manufacturing defect inspection applications
    ''')

    st.markdown('### Convolutional Neural Net (CNN) Basics')
    st.markdown('''
        [Convolutional neural nets](https://towardsdatascience.com/simple-introduction-to-convolutional-neural-networks-cdf8d3077bac) consist of some `n` layers, where each layer is one of the following types:

        - `convolution + ReLU`: Where filters are applied to the original image (this is the crux of the problem).
        - `pooling`: Used to reduce the dimensionality of the network. increases the number of channels to compensate
        - `fully connected + ReLU`: Used to flatten channels of results before classification
        - `softmax`: The final "classification" layer

        Below is an image of an exemplary CNN.
    ''')
    st.image('./assets/descriptive-imgs/other/cnn-basics.png', caption='Basics of a CNN')
    st.markdown('The below image shows intermediate activation maps of VGG-16, a neural net trained for image recognition, which shows a similar low-to-high level of feature extraction from lower to higher layers (ending with a linearly separable classifier).')
    st.image('./assets/descriptive-imgs/other/vgg-16.png', caption='VGG-16 Intermediate Activation Maps')

    st.markdown('### Style Transfer Use Case')
    st.markdown('''
        Below we see the behavior and use case of style transfer. Style Transfer is a technique for transferring style from one image to another.
        For instance, with a neural net trained for style transferral, we can transpose the style of *Starry Night* by Van Gogh to a picture of
        architecture in Tubingen, Germany. [SqueezeNet](https://arxiv.org/abs/1602.07360) is one such neural net trained for style transferral, and it consists of 13 layers, where some
        modules utilize a unique "Fire" architecture, whereby "Fire" modules consist of a "squeeze" layer with 1x1 filters feeding an "expand" layer with 1x1
        and 3x3 filters. This architecture achieves [AlexNet](https://en.wikipedia.org/wiki/AlexNet)-level accuracy on [ImageNet](https://image-net.org/) with 50x fewer parameters.
        The structure is defined below:
    ''')

    st.markdown('### SqueezeNet Feature Summary')
    st.code('''
Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (3): Fire(
        (squeeze): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
        (squeeze_activation): ReLU(inplace=True)
        (expand1x1): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
        (expand1x1_activation): ReLU(inplace=True)
        (expand3x3): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (expand3x3_activation): ReLU(inplace=True)
    )
    (4): Fire(
        (squeeze): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
        (squeeze_activation): ReLU(inplace=True)
        (expand1x1): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
        (expand1x1_activation): ReLU(inplace=True)
        (expand3x3): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (expand3x3_activation): ReLU(inplace=True)
    )
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (6): Fire(
        (squeeze): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
        (squeeze_activation): ReLU(inplace=True)
        (expand1x1): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        (expand1x1_activation): ReLU(inplace=True)
        (expand3x3): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (expand3x3_activation): ReLU(inplace=True)
    )
    (7): Fire(
        (squeeze): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
        (squeeze_activation): ReLU(inplace=True)
        (expand1x1): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        (expand1x1_activation): ReLU(inplace=True)
        (expand3x3): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (expand3x3_activation): ReLU(inplace=True)
    )
    (8): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (9): Fire(
        (squeeze): Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1))
        (squeeze_activation): ReLU(inplace=True)
        (expand1x1): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
        (expand1x1_activation): ReLU(inplace=True)
        (expand3x3): Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (expand3x3_activation): ReLU(inplace=True)
    )
    (10): Fire(
        (squeeze): Conv2d(384, 48, kernel_size=(1, 1), stride=(1, 1))
        (squeeze_activation): ReLU(inplace=True)
        (expand1x1): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
        (expand1x1_activation): ReLU(inplace=True)
        (expand3x3): Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (expand3x3_activation): ReLU(inplace=True)
    )
    (11): Fire(
        (squeeze): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1))
        (squeeze_activation): ReLU(inplace=True)
        (expand1x1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
        (expand1x1_activation): ReLU(inplace=True)
        (expand3x3): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (expand3x3_activation): ReLU(inplace=True)
    )
    (12): Fire(
        (squeeze): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
        (squeeze_activation): ReLU(inplace=True)
        (expand1x1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
        (expand1x1_activation): ReLU(inplace=True)
        (expand3x3): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (expand3x3_activation): ReLU(inplace=True)
    )
)
    ''')
    st.markdown('''
        Fire modules are illustrated in better detail below:
    ''')
    st.image('./assets/descriptive-imgs/other/fire.png', caption='An Illustration of a Fire Module')

    st.markdown('## Feature Visualization')
    st.markdown('''
        - SqueezeNet has 13 different layers and each layer generates an output of a different number of channels (multiple of 16)
        - Features (activation vector for each layer) are represented as tensors
        - To visualize the features of intermediate layers, we capture 16 different channels of the output of every layer depending on an epoch factor selected by a user
        - We experimented with different colormaps including spectral, greyscale, jet, and rainbow
        - We chose spectral because it had a good proportion of Red, Green and Blue values especially for individual pixels in the last few layers
    ''')
    st.markdown('''
        To test intermediate visualization, we used the following content and style image:
    ''')

    col1, col2 = st.columns(2)

    with col1:
        st.image('./assets/input-imgs/tubingen.jpg', caption='A picture of architecture Tubingen, Germany, used as the content image')
    with col2:
        st.image('./assets/input-imgs/starry_night.jpg', caption='A picture of Starry Night by Van Gogh, used as the style image')

    st.markdown('''
        With these input images, we are able to utilize this playground interface to generate intermediate composite images *and* intermediate activation maps of SqueezeNet as ran on them.
        We used the following parameters to generate the intermediate composite images and activation maps:
    ''')

    st.code('''
{
    'content_img': './assets/input-imgs/tubingen.jpg',
    'style_img': './assets/input-imgs/starry_night.jpg',
    'content_size': 192,
    'style_size': 192,
    'style_layers': [1, 4, 6, 7],
    'content_layer': 3,
    'style_weights': [300000, 1000, 15, 3],
    'content_weight': 0.06,
    'tv_weight': 0.02,
    'num_epochs': 200,
    'init_random': False,
    'observe_intermediate_result_count': 5,
    'decay_lr_at': 180,
    'decayed_lr': 0.1,
    'initial_lr': 3.0,
    'layer_vis_choices': [i for i in range(available_layers)],
    'channel_vis_choice': 'first_16',
    'color_mapping': 'nipy_spectral'
}
    ''', language='python')

    st.markdown('Using these hyperparameters, intermediate composite images and activation maps for epochs 0, 100, and 195 (layers 0, 6, and 12 for activation maps) are shown below:')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image('./assets/descriptive-imgs/composite-imgs/image_epoch0.jpeg', caption='An Intermediate Composite Image at Epoch 0')
    with col2:
        st.image('./assets/descriptive-imgs/composite-imgs/image_epoch100.jpeg', caption='An Intermediate Composite Image at Epoch 100')
    with col3:
        st.image('./assets/descriptive-imgs/composite-imgs/image_epoch195.jpeg', caption='An Intermediate Composite Image at Epoch 195')

    st.markdown('**Activation Maps at Epoch 0:**')
    
    st.image('./assets/descriptive-imgs/activation-maps/epoch0/activation_maps_epoch0_layer0.png', caption='An Intermediate Activation Map at Epoch 0, Layer 0')
    st.image('./assets/descriptive-imgs/activation-maps/epoch0/activation_maps_epoch0_layer6.png', caption='An Intermediate Activation Map at Epoch 0, Layer 6')
    st.image('./assets/descriptive-imgs/activation-maps/epoch0/activation_maps_epoch0_layer12.png', caption='An Intermediate Activation Map at Epoch 0, Layer 12')

    st.markdown('**Activation Maps at Epoch 100L**')

    st.image('./assets/descriptive-imgs/activation-maps/epoch100/activation_maps_epoch100_layer0.png', caption='An Intermediate Activation Map at Epoch 100, Layer 0')
    st.image('./assets/descriptive-imgs/activation-maps/epoch100/activation_maps_epoch100_layer6.png', caption='An Intermediate Activation Map at Epoch 100, Layer 6')
    st.image('./assets/descriptive-imgs/activation-maps/epoch100/activation_maps_epoch100_layer12.png', caption='An Intermediate Activation Map at Epoch 100, Layer 12')

    st.markdown('**Activation Maps at Epoch 195:**')

    st.image('./assets/descriptive-imgs/activation-maps/epoch195/activation_maps_epoch195_layer0.png', caption='An Intermediate Activation Map at Epoch 195, Layer 0')
    st.image('./assets/descriptive-imgs/activation-maps/epoch195/activation_maps_epoch195_layer6.png', caption='An Intermediate Activation Map at Epoch 195, Layer 6')
    st.image('./assets/descriptive-imgs/activation-maps/epoch195/activation_maps_epoch195_layer12.png', caption='An Intermediate Activation Map at Epoch 195, Layer 12')

    st.markdown('The final image is shown below.')

    st.image('./assets/descriptive-imgs/composite-imgs/final_img.jpeg', caption='The Final Composite Image')

    st.markdown('## Try it out!')
    st.markdown('''
        To try out this playground, you can input your own images and hyperparameters using the sidebar on the left, or you can simply click the button below to use
        the default images and hyperparameters as defined below:
        - Content Image: Tubingen, Germany (`'tubingen.jpg'`)
        - Style Image: Starry Night by Van Gogh (`'starry_night.jpg'`)
        - Content Image Size: 192
        - Style Image Size: 192
        - Style Layers: [1, 4, 6, 7](#squeezenet-feature-summary)
        - Style Weights: 300000, 1000, 15, 3
        - Content Layer: [3](#squeezenet-feature-summary)
        - Content Weight: 0.06
        - Total Variation Weight: 0.02
        - Number of Epochs: 10
        - Initialize to Random Noise: False
        - Frequency of Observation of Intermediate Result Count: 2
        - Decay Learning Rate At: 9
        - Decayed Learning Rate: 0.10
        - Initial Learning Rate: 3.0
        - Layer Visualization Choices: All Layers
        - Channel Visualization Choices: First 16
        - Colormap: Spectral
    ''')

    st.button(
        'Run Playground with Default Images and Hyperparameters', 
        on_click=lambda: style_transfer, 
        kwargs={
            'content_img': './assets/input-imgs/tubingen.jpg',
            'style_img': './assets/input-imgs/starry_night.jpg'
        }
    )

    with st.sidebar.expander('Table of Contents'):
        st.sidebar.markdown('''
            - [How to Use this Workbench](#how-to-use-this-workbench)
                1. [Upload a Content and Style Image](#step-1-upload-a-content-and-style-image)
                2. [Choose Image Sizes](#step-2-choose-image-sizes)
                3. [Select Style Layers and Weights](#step-3-select-style-layers-and-weights)
                4. [Select Content Layer](#step-4-select-content-layer)
                5. [Select Number of Epochs](#step-5-select-number-of-epochs)
                6. [Select Total Variation Weight](#step-6-select-total-variation-weight)
                7. [Select Learning Rate Hyperparameters](#step-7-select-learning-rate-hyperparameters)
                8. [Modify Output Frequency and Structure of Intermediate Results](#step-8-modify-output-frequency-and-structure-of-intermediate-results)
                9. [Run Style Transfer](#step-9-run-style-transfer) 
            - [Project Introduction](#project-introduction)
                - [Background and Motivation](#background-and-motivation)
                - [Convolutional Neural Net (CNN) Basics](#convolutional-neural-net-cnn-basics)
                - [Style Transfer Use Case](#style-transfer-use-case)
                - [SqueezeNet Feature Summary](#squeezenet-feature-summary)
            - [Feature Visualization](#feature-visualization)
            - [Try it Out!](#try-it-out)
        ''')

    # Sidebar
    st.sidebar.title('Hyperparameter Selection')

    # Sidebar -- Upload Images
    st.sidebar.markdown('## Upload Images')
    st.sidebar.markdown('If you don\'t upload an image, the input images displayed on the dashboard (`tubingen.jpg` and `starry_night.jpg`) will be used.')
    content_img = st.sidebar.file_uploader('Choose a Content Image', type=['jpg'], help='Only JPG images are supported')
    style_img = st.sidebar.file_uploader('Choose a Style Image', type=['jpg'], help='Only JPG images are supported')

    # Display Input Images
    col1, col2 = st.columns(2)

    initial_imgs_container.markdown('## Input Images')

    if content_img is not None:
        with col1:
            initial_imgs_container.image(content_img, caption='Content Image', use_column_width=True)
    else:
        with col1:
            initial_imgs_container.image('./assets/input-imgs/tubingen.jpg', caption='Content Image', use_column_width=True)

    if style_img is not None:
        with col2:
            initial_imgs_container.image(style_img, caption='Style Image', use_column_width=True)
    else:
        with col2:
            initial_imgs_container.image('./assets/input-imgs/starry_night.jpg', caption='Style Image', use_column_width=True)

    # Sidebar -- Upload Image Sizes
    st.sidebar.markdown('## Input Image Sizes')
    content_size = st.sidebar.number_input('Content Image Size', min_value=64, max_value=512, value=192)
    style_size = st.sidebar.number_input('Style Image Size', min_value=64, max_value=512, value=192)

    # Sidebar -- Select Style Layers/Weights to Use
    st.sidebar.markdown('## Style Layers')
    style_layers = st.sidebar.multiselect('Style Layers (no more than 4 recommended)', available_layers, format_func=lambda x: f'Feature {x}', default=[1, 4, 6, 7])
    style_weights = [
        st.sidebar.number_input(f'Style Layer {i} Weight', min_value=1, max_value=500000, value=WEIGHTS_MAP[i] if i in WEIGHTS_MAP else 100) for i in style_layers
    ]

    # Sidebar -- Select Content Layer/Weight to Use
    st.sidebar.markdown('## Content Layer')
    content_layer = st.sidebar.selectbox('Content Layer', available_layers, format_func=lambda x: f'Feature {x}', index=2)
    content_weight = st.sidebar.number_input('Content Layer Weight', min_value=1e-3, max_value=1.0, value=6e-2)

    # Sidebar -- Number of Epochs
    st.sidebar.markdown('## Number of Epochs')
    num_epochs = st.sidebar.number_input('Number of Epochs', min_value=1, max_value=500, value=200)

    # Sidebar -- TV
    st.sidebar.markdown('## Total Variation')
    tv_weight = st.sidebar.number_input('Total Variation Weight', min_value=1e-3, max_value=1.0, value=2e-2)

    # Sidebar -- Learning Rate
    st.sidebar.markdown('## Learning Rate Hyperparameters')
    decay_lr_at = st.sidebar.number_input('Decay Learning Rate At', min_value=1, max_value=num_epochs, value=int(0.9 * num_epochs))
    decayed_lr = st.sidebar.number_input('Decayed Learning Rate', min_value=0.1, max_value=1.0, value=0.1)
    initial_lr = st.sidebar.number_input('Initial Learning Rate', min_value=1.0, max_value=5.0, value=3.0)

    # Sidebar -- Intermediate Vis
    st.sidebar.markdown('## Intermediate Visualization')
    layer_vis_choices = st.sidebar.multiselect('Intermediate Layers to Visualize', available_layers, format_func=lambda x: f'Feature {x}', default=[i for i in available_layers])
    channel_vis_choice = st.sidebar.selectbox('Channels to Visualize', available_vis_channel_methods, index=0)
    observe_intermediate_result_count = st.sidebar.number_input('Epoch Frequency for Observing Intermediate Results', min_value=1, max_value=20, value=5)
    color_mapping = st.sidebar.selectbox('Color Mapping', ['nipy_spectral', 'jet', 'gray', 'rainbow'], index=0)
    init_random = st.sidebar.checkbox('Random Initialization', value=False)

    args = {
        'content_img': content_img,
        'style_img': style_img,
        'content_size': content_size,
        'style_size': style_size,
        'style_layers': style_layers,
        'content_layer': content_layer,
        'style_weights': style_weights,
        'content_weight': content_weight,
        'tv_weight': tv_weight,
        'num_epochs': num_epochs,
        'init_random': init_random,
        'observe_intermediate_result_count': observe_intermediate_result_count,
        'decay_lr_at': decay_lr_at,
        'decayed_lr': decayed_lr,
        'initial_lr': initial_lr,
        'layer_vis_choices': layer_vis_choices,
        'channel_vis_choice': channel_vis_choice,
        'color_mapping': color_mapping
    }

    st.sidebar.button(
        'Run Style Transfer', 
        key='run_style_transfer', 
        on_click=run_style_transfer,
        kwargs=args
    )


if __name__ == "__main__":
    for param in CNN.parameters():
        param.requires_grad_(False)

    remove_all_previous_figs()

    main()
