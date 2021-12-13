from logging import info
import streamlit as st
from streamlit.elements.legacy_data_frame import add_rows
from streamlit.logger import init_tornado_logs
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import PIL
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406])
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225])
DTYPE = torch.cuda.FloatTensor

CNN = torchvision.models.squeezenet1_1(pretrained=True).features
CNN.type(DTYPE)

WEIGHTS_MAP = {
    1: 300000,
    4: 1000,
    6: 15,
    7: 3
}

def preprocess_image(img, size=512):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])

    return transform(img)


def deprocess_image(img):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in SQUEEZENET_STD.tolist()]),
        T.Normalize(mean=[-m for m in SQUEEZENET_MEAN.tolist()], std=[1, 1, 1]),
        T.Lambda(rescale),
        T.ToPILImage(),
    ])

    return transform(img)


def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


def features_from_img(imgpath, imgsize):
    img = preprocess_image(PIL.Image.open(imgpath), size=imgsize)
    img_var = img.type(DTYPE)
    return extract_features(img_var), img_var
    

def extract_features(x, cnn=CNN):
    features = []
    prev_feat = x

    for i, module in enumerate(cnn._modules.values()):
        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    
    return features


def content_loss(content_weight, content_current, content_original):
    shape = content_current.shape
    print('content shape: ', shape)
    dim = shape[1] * shape[2] * shape[3]

    F = content_current.reshape(dim)
    P = content_original.reshape(dim)

    lc = sum(((F - P) ** 2))
    lc *= content_weight

    return lc


def gram_matrix(features, normalize=True):
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
    image = img
    image1 = image[:, :, 1:, :].flatten()
    image2 = image[:, :, :-1, :].flatten()
    image3 = image[:, :, :, 1:].flatten()
    image4 = image[:, :, :, :-1].flatten()

    return tv_weight * (sum((image2 - image1) ** 2) + sum((image4 - image3) ** 2))


# def change_color_mapping(fig, cmap):
#     for ax in fig.axes:
#         ax.set_cmap(cmap)


def download_all_figs(figs, epoch):
    for i, fig in enumerate(figs):
        fig.savefig(f'activation_maps_{epoch}_{i}.png')


def layer_vis(feats, num_epoch, output_container, color_mapping):
    activation_container = output_container.expander(f'Activation Maps for Epoch {num_epoch}')
    figs = []

    for num_layer in range(len(feats)):
        fig, axes = plt.subplots(2, 8, figsize=(50, 10))
        fig.suptitle(f'Activation Maps for Layer {num_layer}', fontsize=36)

        layer_vis = feats[num_layer][0, :, :, :].data.cpu()
        row = 0
        
        for i, filter in enumerate(layer_vis):
            if i == 16:
                break
            
            if i == 8:
                row = 1
            
            axes[row, i % 8].axis('off')
            axes[row, i % 8].imshow(filter, cmap=color_mapping)

        activation_container.pyplot(fig)
        figs.append(fig)

    # activation_container.button('Download Activation Maps', on_click=download_all_figs, args=[figs, num_epoch])


def style_transfer(
    content_img, 
    style_img, 
    content_size, 
    style_size,
    style_layers, 
    content_layer,
    content_weight, 
    style_weights,
    tv_weight,
    initial_lr,
    decayed_lr,
    decay_lr_at,
    num_epochs=200,
    init_random=False,
    observe_intermediate_result_count=5,
    color_mapping='nipy_spectral'
):
    import time

    try:
        start_time = time.time()
        output_container = st.container()

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
        exception_message = output_container.empty()
        info_message = output_container.info('Starting style transfer...')
        img_container = output_container.expander('Image Outputs')
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
                
                layer_vis(feats, t, output_container, color_mapping)
                # st.image(deprocess_image(img.data.cpu()), caption=f'Image at Epoch {t + 1}')

            info_message.info(f'Epoch {t} completed. Total elapsed time: {round(time.time() - epoch_start_time, 2)}s')

        output_container.markdown('## Final Image')
        output_container.image(deprocess_image(img.data.cpu()), caption='Final Image')
        
        # fig, ax = plt.subplots()
        # ax.axis('off')
        # ax.imshow(deprocess_image(img.data.cpu()))
        # st.pyplot(fig)
        # st.download_button('Download Final Image', deprocess_image(img.data.cpu()))
        
        st.balloons()
        info_message.success(f'Finished! Total elapsed time: {round(time.time() - start_time, 2)}s')
    except Exception as e:
        exception_message.exception(e)


def run_style_transfer(**args):
    for key in args:
        if args[key] is None:
            print(key)
            st.error(f'Missing fields: {[key for key in args if args[key] is None]}')
            st.error('Please fill in all fields')
            
            return

    style_transfer(**args)
    

def main():
    """
    Main function
    """
    st.set_page_config(page_title='Style Transfer Vis', page_icon=':art:')

    initial_imgs_container = st.container()
    descriptive_container = st.container()

    descriptive_container.title('Style Transfer Neural Network Visualization')
    descriptive_container.markdown('## Jared Amen, Pranav Rajan, and Alan Weber')
    descriptive_container.markdown('## CS 6965: Advanced Data Visualization -- Professor Bei Wang-Phillips -- University of Utah -- Fall 2021')

    st.sidebar.title('Hyperparameter Selection')
    available_layers = [i for i in range(1, len(CNN) + 1)]

    st.sidebar.markdown('## Upload Images')
    content_img = st.sidebar.file_uploader('Choose a Content Image', type=['jpg'], help='Only JPG images are supported')
    style_img = st.sidebar.file_uploader('Choose a Style Image', type=['jpg'], help='Only JPG images are supported')

    col1, col2 = initial_imgs_container.columns(2)

    if content_img is not None:
        initial_imgs_container.markdown('## Input Images')
        with col1:
            initial_imgs_container.image(content_img, caption='Content Image', use_column_width=True)

    if style_img is not None:
        with col2:
            initial_imgs_container.image(style_img, caption='Style Image', use_column_width=True)

    st.sidebar.markdown('## Input Image Sizes')
    content_size = st.sidebar.number_input('Content Image Size', min_value=64, max_value=512, value=192)
    style_size = st.sidebar.number_input('Style Image Size', min_value=64, max_value=512, value=192)

    st.sidebar.markdown('## Style Layers')
    style_layers = st.sidebar.multiselect('Style Layers (no more than 4 recommended)', available_layers, format_func=lambda x: f'Feature {x}', default=[1, 4, 6, 7])
    style_weights = [
        st.sidebar.number_input(f'Style Layer {i} Weight', min_value=1, max_value=500000, value=WEIGHTS_MAP[i] if i in WEIGHTS_MAP else 100) for i in style_layers
    ]

    st.sidebar.markdown('## Content Layer')
    content_layer = st.sidebar.selectbox('Content Layer', available_layers, format_func=lambda x: f'Feature {x}', index=2)
    content_weight = st.sidebar.number_input('Content Layer Weight', min_value=1e-3, max_value=1.0, value=6e-2)

    st.sidebar.markdown('## Other')
    tv_weight = st.sidebar.number_input('Total Variation Weight', min_value=1e-3, max_value=1.0, value=2e-2)
    num_epochs = st.sidebar.number_input('Number of Epochs', min_value=1, max_value=500, value=200)
    init_random = st.sidebar.checkbox('Random Initialization', value=False)
    decay_lr_at = st.sidebar.number_input('Decay Learning Rate At', min_value=1, max_value=num_epochs, value=int(0.9 * num_epochs))
    decayed_lr = st.sidebar.number_input('Decayed Learning Rate', min_value=0.1, max_value=1.0, value=0.1)
    initial_lr = st.sidebar.number_input('Initial Learning Rate', min_value=1.0, max_value=5.0, value=3.0)
    observe_intermediate_result_count = st.sidebar.number_input('Epoch Frequency for Observing Intermediate Results', min_value=1, max_value=20, value=5)
    color_mapping = st.sidebar.selectbox('Color Mapping', ['nipy_spectral', 'jet', 'gray', 'rainbow'], index=0)

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

    main()
