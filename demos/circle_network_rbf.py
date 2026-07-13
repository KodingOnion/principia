import random

import matplotlib.pyplot as plt
import numpy as np

from principia import mse_loss
from principia import AdamOptimiser
from principia import Sequential
from principia import Tensor
from principia import RBFLayer

def circle_generator(num_points):
    inputs = []
    outputs = []

    for _ in range(num_points):
        x = random.uniform(-1.0, 1.0)
        y = random.uniform(-1.0, 1.0)
        dist = (x**2 + y**2) ** 0.5
        output = 0 if (dist > 0.6 or dist < 0.2) else 1

        inputs.append([x, y])
        outputs.append([output])

    return inputs, outputs


def main(num_points=1000, learning_rate=0.01, epochs=1000, plot_interval=10, batch_size=12,show_plot=True):
    x, y = circle_generator(num_points)

    x = Tensor(np.array(x))
    y = Tensor(np.array(y).reshape(-1, 1))


    model = Sequential([RBFLayer(2,8,12),RBFLayer(8,8,12),RBFLayer(8,1,12)])
    optimiser = AdamOptimiser(model.parameters(), learning_rate)

    xx = np.linspace(-1, 1, 50)
    yy = np.linspace(-1, 1, 50)
    xx_mesh, yy_mesh = np.meshgrid(xx, yy)
    input_mesh = np.stack([xx_mesh.flatten(), yy_mesh.flatten()], axis=1)
    input_tensor = Tensor(input_mesh)

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal", adjustable="box")
    cbar = None
    plt.show(block=False)

    for i in range(1, epochs + 1):

        for j in range(0, len(x.data), batch_size):
            x_batch = Tensor(x.data[j:j+batch_size])
            y_batch = Tensor(y.data[j:j+batch_size])

            predictions = model(x_batch)
            mse = mse_loss(predictions, y_batch)

            model.zero_grad()
            mse.backward()
            optimiser.step()

        if i == 1 or i % plot_interval == 0:
            print(f"EPOCH NUM:{i} LOSS:{mse.data}")

            mesh_preds = model(input_tensor)
            final_map = mesh_preds.reshape((50, 50)).data

            if cbar is not None:
                cbar.remove()

            ax.clear()

            contour = ax.contourf(xx, yy, final_map, levels=20, vmin=-1, vmax=1, cmap="coolwarm")
            
            cbar = fig.colorbar(contour, ax=ax, label='Network Output')
            
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"Free-RBF-KAN Decision Boundary - Epoch {i}")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.01)

    if show_plot:
        plt.ioff()
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()