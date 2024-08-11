Open in colab 
upload the relevant dataset 
run the code
main algorithm
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.interpolate import splprep, splev
import svgwrite
import cairosvg

# Function to read CSV and parse paths
def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

# Function to check if a shape is a straight line
def is_straight_line(XY, tolerance=1e-2):
    if len(XY) < 2:
        return False
    x0, y0 = XY[0]
    x1, y1 = XY[-1]
    distances = np.abs((x1 - x0) * (y0 - XY[:, 1]) - (x0 - XY[:, 0]) * (y1 - y0)) / distance.euclidean((x0, y0), (x1, y1))
    return np.all(distances < tolerance)

# Function to check if a shape is a circle
def is_circle(XY, tolerance=1e-2):
    if len(XY) < 3:
        return False
    center = np.mean(XY, axis=0)
    radii = np.linalg.norm(XY - center, axis=1)
    return np.std(radii) < tolerance

# Function to classify shapes into regular shapes
def regularize_curves(paths_XYs):
    regular_shapes = []
    for XYs in paths_XYs:
        for XY in XYs:
            if is_straight_line(XY):
                regular_shapes.append(('line', XY))
            elif is_circle(XY):
                regular_shapes.append(('circle', XY))
    return regular_shapes

# Function to check if a shape is symmetric
def is_symmetric(XY, tolerance=1e-2):
    mid_x = (np.max(XY[:, 0]) + np.min(XY[:, 0])) / 2
    reflected_XY = np.copy(XY)
    reflected_XY[:, 0] = 2 * mid_x - reflected_XY[:, 0]
    return np.mean(np.min(distance.cdist(XY, reflected_XY), axis=1)) < tolerance

# Function to detect symmetry in shapes
def detect_symmetry(paths_XYs):
    symmetric_shapes = []
    for XYs in paths_XYs:
        for XY in XYs:
            if is_symmetric(XY):
                symmetric_shapes.append(XY)
    return symmetric_shapes

# Function to complete a curve by interpolation
def complete_curve(XY, num_points=100):
    tck, _ = splprep([XY[:, 0], XY[:, 1]], s=0)
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.vstack([x_new, y_new]).T

# Function to complete curves in the paths
def complete_curves(paths_XYs):
    completed_shapes = []
    for XYs in paths_XYs:
        for XY in XYs:
            completed_XY = complete_curve(XY)
            completed_shapes.append(completed_XY)
    return completed_shapes

# Function to plot paths
def plot(paths_XYs, ax, title="Paths", color='blue'):
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            ax.plot(XY[:, 0], XY[:, 1], color=color, linewidth=2)
    ax.set_aspect('equal')
    ax.set_title(title)

# Utility function to plot shapes
def plot_shapes(shapes, title, ax, color='blue'):
    for shape, XY in shapes:
        ax.plot(XY[:, 0], XY[:, 1], color=color, label=shape, linewidth=2)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend()

# Function to save paths as SVG and PNG
def polylines2svg(paths_XYs, svg_path):
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            if XY.shape[1] < 2:
                print(f"Unexpected shape in XY: {XY.shape}")
                continue
            W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)
    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
    colours = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']

    group = dwg.g()
    for i, path in enumerate(paths_XYs):
        path_data = []
        c = colours[i % len(colours)]
        for XY in path:
            if XY.shape[1] < 2:
                print(f"Skipping malformed XY: {XY}")
                continue
            path_data.append(("M", (XY[0, 0], XY[0, 1])))
            for j in range(1, len(XY)):
                path_data.append(("L", (XY[j, 0], XY[j, 1])))
            if not np.allclose(XY[0], XY[-1]):
                path_data.append(("Z", None))
        group.add(dwg.path(d=path_data, fill=c, stroke='none', stroke_width=2))
    dwg.add(group)
    dwg.save()

    png_path = svg_path.replace('.svg', '.png')
    fact = max(1, 1024 // min(H, W))
    cairosvg.svg2png(url=svg_path, write_to=png_path, parent_width=W, parent_height=H, output_width=fact * W, output_height=fact * H, background_color='white')

# Main function with visualization
def main():
    csv_path = 'frag0.csv'
    paths_XYs = read_csv(csv_path)

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Plot original paths
    plot(paths_XYs, axs[0, 0], "Original Paths")

    # Check and plot for regular shapes
    regular_shapes = regularize_curves(paths_XYs)
    if regular_shapes:
        plot_shapes(regular_shapes, "Regularized Shapes", axs[0, 1], color='green')

    # Check and plot for symmetry
    symmetric_shapes = detect_symmetry(paths_XYs)
    if symmetric_shapes:
        plot_shapes([('symmetric', XY) for XY in symmetric_shapes], "Symmetric Shapes", axs[1, 0], color='red')

    # Check and plot for curve completion
    completed_shapes = complete_curves(paths_XYs)
    plot_shapes([('completed', XY) for XY in completed_shapes], "Completed Curves", axs[1, 1], color='purple')

    plt.tight_layout()
    plt.show()

    # Save to SVG and PNG
    output_svg_path = 'output.svg'
    polylines2svg(paths_XYs, output_svg_path)
    print(f"SVG and PNG files saved to {output_svg_path}")

if __name__ == '__main__':
    main()
