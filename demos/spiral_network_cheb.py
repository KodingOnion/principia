import random

import matplotlib.pyplot as plt
import numpy as np

from principia import mse_loss
from principia import AdamOptimiser
from principia import Sequential
from principia import Tensor
from principia import ChebLayer
from principia import RBFLayer

import time

start = time.time()

def spiral_gen(n_points, noise, revolutions):
    angles = np.random.rand(n_points)
    angles = np.sqrt(angles)
    theta = angles * (revolutions * 2 * np.pi)

    r = theta
    
    X_a = r * np.cos(theta) + (np.random.randn(n_points) * noise)
    Y_a = r * np.sin(theta) + (np.random.randn(n_points) * noise)

    X_b = -r * np.cos(theta) + (np.random.randn(n_points) * noise)
    Y_b = -r * np.sin(theta) + (np.random.randn(n_points) * noise)

    coords_a = np.column_stack((X_a, Y_a))
    coords_b = np.column_stack((X_b, Y_b))

    coords = np.vstack((coords_a, coords_b))

    max_val = np.max(np.abs(coords))
    coords = coords / max_val

    zeros_array = np.zeros(n_points)
    ones_array = np.ones(n_points)

    labels = np.concatenate((zeros_array, ones_array))

    return coords,labels

def main(num_points=1000, noise=0.5, revolutions=3, learning_rate=0.01, epochs=1000, plot_interval=10, batch_size=12,show_plot=True):
    x, y = spiral_gen(num_points, noise, revolutions)

    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    x = Tensor(np.array(x))
    y = Tensor(np.array(y).reshape(-1, 1))

    model = Sequential([ChebLayer(2, 8, 6), ChebLayer(8, 8, 6), ChebLayer(8, 1, 6)])
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
            time_taken = time.time() - start
            time_taken = round(time_taken,2)
            print(f"EPOCH NUM:{i} LOSS:{mse.data} TIME:{time_taken}")

            mesh_preds = model(input_tensor)
            final_map = mesh_preds.reshape((50, 50)).data

            if cbar is not None:
                cbar.remove()

            ax.clear()

            contour = ax.contourf(xx, yy, final_map, levels=20, vmin=0, vmax=1, cmap="coolwarm")
            ax.scatter(x.data[:, 0], x.data[:, 1], c=y.data.flatten(), cmap="bwr", edgecolors="black", alpha=0.5, s=20)
            
            cbar = fig.colorbar(contour, ax=ax, label='Network Output')
            
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"ChebKAN Decision Boundary - Epoch {i}")

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