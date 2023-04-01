from PoissonEditing import *

def read_image(clone):
    allowed_extensions = ['.jpg', '.png', '.jpeg']
    source_directory = r"images/source.jpg"
    source = cv2.imread(source_directory)
    dest = 0

    if clone:
        dest_directory = r"images/dest.jpg"
        dest = cv2.imread(dest_directory)
    return source, dest


def main():
    clone = True
    grad_mix = True

    temp, dest = read_image(clone)
    source = temp.copy()
    pe = PoissonEditing(temp, dest)
    points, mask = pe.get_mask()

    if clone:
        # save image
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(source)
        axs[0].set_title('source')
        axs[1].imshow(dest)
        axs[1].set_title('destination')
        axs[2].imshow(mask, cmap="gray")
        axs[2].set_title('mask')

        plt.savefig("mask.png")
    else:
        # save image
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(source)
        axs[0].set_title('source')
        axs[1].imshow(mask)
        axs[1].set_title('mask')

        plt.savefig("mask.png")

    image = pe.colour(mask, clone, grad_mix)

    # save image
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(source)
    axs[0].set_title('source')
    axs[1].imshow(dest)
    axs[1].set_title('destination')
    axs[2].imshow(image)
    axs[2].set_title('editted')

    plt.savefig("output.png")

if __name__=="__main__":
    main()
