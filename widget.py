import ipywidgets as widgets
from ipywidgets import interactive
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def conv_simulation(input_size, filter_size, padding, stride):
    img = Image.open("/Users/imb/2024/ironhack/data_curriculum/ai_assignment/img/cat.png").convert('L')
    img = img.resize((input_size, input_size))
    img_array = np.array(img)

    filter = np.ones((filter_size, filter_size))
    padded_input = np.pad(img_array, padding, mode='constant')
    output_size = (input_size + 2 * padding - filter_size) // stride + 1
    output_matrix = np.zeros((output_size, output_size))

    for i in range(output_size):
        for j in range(output_size):
            output_matrix[i, j] = np.sum(
                padded_input[i*stride:i*stride+filter_size, j*stride:j*stride+filter_size]
            )

    output_matrix = (output_matrix - output_matrix.min()) / (output_matrix.max() - output_matrix.min())

    fig, axs = plt.subplots(1, 2, figsize=(4, 4))

    axs[0].imshow(padded_input, cmap='gray')
    axs[0].set_title("Padded Input Image", fontsize=10)
    axs[0].axis('off')
    
    axs[1].imshow(output_matrix, cmap='gray')
    axs[1].set_title("Convolution Output", fontsize=10)
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()
    
def run_conv_simulation():
    interactive_conv = interactive(conv_simulation, 
                                   input_size=widgets.IntSlider(min=28, max=512, step=32, value=128), 
                                   filter_size=widgets.IntSlider(min=2, max=10, step=1, value=3), 
                                   stride=widgets.IntSlider(min=1, max=5, step=1, value=1),
                                   padding=widgets.IntSlider(min=0, max=20, step=1, value=0)
                                  )
    display(interactive_conv)
