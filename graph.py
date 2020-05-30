import matplotlib.pyplot as plt


class Graph:
    def __init__(self, q_table, save_plot_path, save_plot=True):
        self.q_table = q_table
        self.save_plot_path = save_plot_path
        self.save_plot = save_plot

    def get_q_color(self, value, vals):
        if value == max(vals):
            return "green", 1.0
        else:
            return "red", 0.3

    def plot(self):
        fig = plt.figure(figsize=(12, 9))

        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        for x, x_vals in enumerate(self.q_table):
            for y, y_vals in enumerate(x_vals):
                ax1.scatter(x, y, c=self.get_q_color(y_vals[0], y_vals)[0], marker="o",
                            alpha=self.get_q_color(y_vals[0], y_vals)[1])
                ax2.scatter(x, y, c=self.get_q_color(y_vals[1], y_vals)[0], marker="o",
                            alpha=self.get_q_color(y_vals[1], y_vals)[1])
                ax3.scatter(x, y, c=self.get_q_color(y_vals[2], y_vals)[0], marker="o",
                            alpha=self.get_q_color(y_vals[2], y_vals)[1])

                ax1.set_ylabel("Action 0")
                ax2.set_ylabel("Action 1")
                ax3.set_ylabel("Action 2")

        if self.save_plot:
            plt.savefig(self.save_plot_path)

        plt.show()
        return
