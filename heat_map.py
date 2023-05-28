import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable
def read_grayscale_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    # return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)[200:310,300:400]


def smooth_image(image, sigma=1.0):
    return gaussian_filter(image, sigma)

def draw_arrow():
    img = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute gradients in x and y directions
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)

    # Compute gradient magnitude and direction
    magnitude = np.sqrt(sobelx**2.0 + sobely**2.0)
    direction = np.arctan2(sobely, sobelx)

    # Parameters for quiver plot
    step = 10
    x = np.arange(0, img.shape[1], step)
    y = np.arange(0, img.shape[0], step)
    X, Y = np.meshgrid(x, y)

    # Compute averaged direction in local neighborhood
    direction_avg = np.zeros_like(direction)
    for i in range(0, direction.shape[0], step):
        for j in range(0, direction.shape[1], step):
            direction_avg[i:i+step, j:j+step] = np.mean(direction[i:i+step, j:j+step])

    U = np.cos(direction_avg)[::step, ::step]
    V = np.sin(direction_avg)[::step, ::step]

    # Plot the original image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Plot the quiver plot of gradient directions
    plt.quiver(X, Y, U, V, color='r', angles='xy', scale_units='xy',)

    plt.gca()  # Invert y axis to match the image coordinate system
    # plt.show()
    plt.savefig('arrow.jpg')
def visualize_height(image, smoothing_sigma=1.0):
    # image = smooth_image(image, smoothing_sigma)
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    vector_x = np.cos(angle * np.pi / 180) * magnitude
    vector_y = np.sin(angle * np.pi / 180) * magnitude
    vector_z = np.zeros_like(magnitude)
    # 计算梯度的幅值
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # image = grad
    image = smooth_image(image, smoothing_sigma)
    x = np.arange(0, image.shape[1])
    y = np.arange(0, image.shape[0])
    x, y = np.meshgrid(x, y)
    image = 4*image/75+27
    fig = plt.figure()
    # ax = fig.gca(projection='3d')

    ax = fig.add_axes(Axes3D(fig))   

    # 使用自定义的颜色映射
    colormap = plt.get_cmap('coolwarm')

    # 根据灰度值归一化，并映射到0-1之间
    norm = plt.Normalize(image.min(), image.max())
    colors = colormap(norm(image))

    surf = ax.plot_surface(x, y, image, rstride=1, cstride=1, facecolors=colors, linewidth=0, antialiased=False, alpha=1.0)

    # 添加等高线
    contours = 10
    ax.contour(x, y, image, contours, colors='black', linewidths=1)
    sm = ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, shrink=0.5, aspect=5)
    cbar.set_label('Temperature')
    plt.savefig('4.png')

if __name__ == "__main__":
    file_path = "test.jpg"  # 替换为你的灰度图像路径
    smoothing_sigma = 2.0  # 设置平滑参数，可根据需求调整
    img = cv2.imread(file_path)
    draw_arrow()
    # gray_image = read_grayscale_image(file_path)
    # visualize_height(gray_image, smoothing_sigma)
