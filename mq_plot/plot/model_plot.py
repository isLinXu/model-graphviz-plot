# import matplotlib.pyplot as plt
# import numpy as np
# import timm
#
# model_name = 'vit_base_patch16_224'
# model = timm.create_model(model_name, pretrained=True)
#
# # Define the figure size and axis
# fig = plt.figure(figsize=(20, 10))
# ax = fig.add_subplot(111)
#
# # Define the node size and color
# node_size = 800
# node_color = 'lightblue'
#
# # Define the edge color and width
# edge_color = 'gray'
# edge_width = 0.5
#
# # Define the position of each node
# pos = {}
# offset = 0
# for name, module in model.named_children():
#     pos[name] = (0, offset)
#     offset -= 1
#
# # Draw the nodes
# for name, module in model.named_children():
#     ax.scatter(pos[name][0], pos[name][1], s=node_size, c=node_color, edgecolors='black', linewidths=0.5)
#     ax.annotate(name, xy=pos[name], xytext=(-20, 0), textcoords='offset points', ha='right', va='center')
#
#     # Draw the edges
#     for child_name, child_module in module.named_children():
#         if child_name not in pos:
#             pos[child_name] = (1, 0)
#         ax.plot([pos[name][0], pos[child_name][0]], [pos[name][1], pos[child_name][1]], color=edge_color, linewidth=edge_width)
#
# # Set the axis limits and remove the ticks
# ax.set_xlim([-1, 2])
# ax.set_ylim([offset+1, 1])
# ax.set_xticks([])
# ax.set_yticks([])
#
# # Save the figure
# plt.savefig('vit_base_patch16_224.png', dpi=300, bbox_inches='tight')

import matplotlib.pyplot as plt
import numpy as np
import timm
from torch import nn

model_name = 'vit_base_patch16_224'
model = timm.create_model(model_name, pretrained=True)

# Define the figure size and axis
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)

# Define the node size and color
node_size = 800
node_color = 'lightblue'

# Define the edge color and width
edge_color = 'gray'
edge_width = 0.5

# Define the position of each node
pos = {}
offset = 0
for name, module in model.named_children():
    pos[name] = (0, offset)
    offset -= 1

# Draw the nodes
for name, module in model.named_children():
    ax.scatter(pos[name][0], pos[name][1], s=node_size, c=node_color, edgecolors='black', linewidths=0.5)
    ax.annotate(name, xy=pos[name], xytext=(-20, 0), textcoords='offset points', ha='right', va='center')

    # Draw the edges
    for child_name, child_module in module.named_children():
        if isinstance(child_module, (nn.Sequential, nn.ModuleList)):
            for i, subchild_module in enumerate(child_module):
                subchild_name = f"{child_name}_{i}"
                pos[subchild_name] = (pos[name][0]+1, pos[name][1]-i)
                ax.scatter(pos[subchild_name][0], pos[subchild_name][1], s=node_size, c=node_color, edgecolors='black', linewidths=0.5)
                ax.annotate(subchild_name, xy=pos[subchild_name], xytext=(20, 0), textcoords='offset points', ha='left', va='center')
                ax.plot([pos[name][0], pos[subchild_name][0]], [pos[name][1], pos[subchild_name][1]], color=edge_color, linewidth=edge_width)
        else:
            pos[child_name] = (pos[name][0]+1, pos[name][1])
            ax.scatter(pos[child_name][0], pos[child_name][1], s=node_size, c=node_color, edgecolors='black', linewidths=0.5)
            ax.annotate(child_name, xy=pos[child_name], xytext=(20, 0), textcoords='offset points', ha='left', va='center')
            ax.plot([pos[name][0], pos[child_name][0]], [pos[name][1], pos[child_name][1]], color=edge_color, linewidth=edge_width)

# Set the axis limits and remove the ticks
ax.set_xlim([-1, 3])
ax.set_ylim([offset+1, 1])
ax.set_xticks([])
ax.set_yticks([])

# Save the figure
plt.savefig('vit_base_patch16_224.png', dpi=300, bbox_inches='tight')