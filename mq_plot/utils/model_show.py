

import timm
import matplotlib.pyplot as plt

def draw_model(model, input_size, hidden_size, output_size):
    # Print the model summary
    print(model)
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(20, 20))

    # Define the node sizes
    input_size = input_size / 1000

    # Draw the input node and label
    ax.add_patch(plt.Circle((0, 0), radius=input_size, color='green'))
    ax.text(0, 0, f'Input\n({input_size*1000}x{input_size*1000}x3)', fontsize=20, ha='center', va='center')

    # Draw the hidden nodes and connections
    layer_count = 1
    for layer in model.children():
        # Draw the node
        ax.add_patch(plt.Circle((layer_count, 0), radius=hidden_size, color='blue'))
        # Draw the connection
        ax.plot([layer_count-1, layer_count], [0, 0], 'k-', lw=2)
        # Add the layer label
        layer_name = str(layer).split('(')[0]
        ax.text(layer_count, 0, layer_name, fontsize=16, ha='center', va='center')
        layer_count += 1

    # Draw the output node and label
    ax.add_patch(plt.Circle((layer_count, 0), radius=output_size, color='red'))
    ax.text(layer_count, 0, f'Output\n({model.num_classes})', fontsize=20, ha='center', va='center')

    # Set the axis limits and remove the ticks
    ax.set_xlim(-1, layer_count+1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    # Show the plot
    plt.show()

if __name__ == '__main__':
    # Load the model
    model_name = 'vit_base_patch16_224'
    model = timm.create_model(model_name, pretrained=True)
    input_size = 224
    hidden_size = 0.3
    output_size = 0.5
    # Draw the model
    draw_model(model, input_size, hidden_size, output_size)

