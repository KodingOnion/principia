from principia import AdamOptimiser, KAN, Tensor, mse_loss
import matplotlib.pyplot as plt
import numpy as np
import random


def circle_generator(num_points):
    inputs = []
    outputs = []

    for _ in range(num_points):
        x = random.uniform(-1.0, 1.0)
        y = random.uniform(-1.0, 1.0)
        dist = (x**2 + y**2) ** 0.5
        output = 0 if dist > 0.6 else 1

        inputs.append([x, y])
        outputs.append([output])

    return inputs, outputs


def main(num_points=10000, learning_rate=0.01, epochs=100000, plot_interval=10, show_plot=True):
    x, y = circle_generator(num_points)

    x = Tensor(np.array(x))
    y = Tensor(np.array(y).reshape(-1, 1))

    model = KAN([2, 12, 1], 24)
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
    plt.show(block=False)

    for i in range(1, epochs + 1):
        predictions = model(x)
        mse = mse_loss(predictions, y)

        model.zero_grad()
        mse.backward()
        optimiser.step()

        if i == 1 or i % plot_interval == 0:
            print(f"EPOCH NUM:{i} LOSS:{mse.data}")

            mesh_preds = model(input_tensor)
            final_map = mesh_preds.reshape((50, 50)).data

            ax.clear()
            ax.contourf(xx, yy, final_map, levels=20, vmin=-1, vmax=1, cmap="coolwarm")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"KAN Decision Boundary - Epoch {i}")

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