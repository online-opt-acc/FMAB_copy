def savefig(fig, path, name):
    fig.tight_layout()
    # fig.savefig(str(path / f"{name}_image.png"))
    fig.savefig(str(path / f"{name}_image.pdf"))

    # data = np.array(fig.canvas.buffer_rgba())
    # weights = [0.2989, 0.5870, 0.1140]
    # data = np.dot(data[..., :-1], weights)
    # plt.imsave(str(path / f"{name}_image_gray.png"), data, cmap="gray")
    # plt.imsave(str(path / f"{name}_image_gray.pdf"), data, cmap="gray")
