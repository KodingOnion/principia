"""Train a KAN model to approximate a sine wave and plot results dynamically."""

from principia import AdamOptimiser, KAN, Tensor, mse_loss
import numpy as np
import matplotlib.pyplot as plt

def main(learning_rate=0.01, epochs=2500, plot_interval=50, show_plot=True):
    X_data = np.linspace(-np.pi, np.pi, 100).reshape(100, 1)
    Y_data = np.sin(X_data)
    X = Tensor(X_data)
    Y = Tensor(Y_data)

    model = KAN(layer_sizes=[1, 5, 1], num_centers=10)
    optimiser = AdamOptimiser(model.parameters(), learning_rate=learning_rate)

    plt.ion()
    fig, ax = plt.subplots()

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("KAN Sine Wave Approximation - Epoch 0")

    ax.plot(X_data, Y_data, label="True Sine Wave", color="blue", linewidth=2)

    pred_line, = ax.plot(X_data, np.zeros_like(X_data), label="KAN Approximation", color="red", linestyle="dashed", linewidth=2)
    ax.legend()
    plt.show(block=False)

    for i in range(1, epochs + 1):
        predictions = model(X)
        MSE = mse_loss(predictions, Y)

        model.zero_grad()
        MSE.backward()
        optimiser.step()

        if i == 1 or i % plot_interval == 0:
            print(f"EPOCH NUM:{i} LOSS:{MSE.data}")

            pred_line.set_ydata(predictions.data)
            ax.set_title(f"KAN Sine Wave Approximation - Epoch {i} (Loss: {MSE.data:.5f})")

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