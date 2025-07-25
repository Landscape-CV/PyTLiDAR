from RobustCylinderFitting import RobustCylinderFitterEcomodel
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import rpca.ialm  # Needed for 3D plotting
from sklearn.covariance import MinCovDet 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def load_point_cloud(file_path, intensity_threshold = 0, full_data = False):
    """
    Load a point cloud from LAS or LAZ files.

    Parameters:
    file_path : str
        Path to the LAS or LAZ file.

    Returns:
    point_cloud : ndarray
        Nx3 matrix of point coordinates (x, y, z).
    """
    if ".xyz" in file_path:
        # Load point cloud from an XYZ file
        point_data = np.loadtxt(file_path, dtype=np.float64)
        if point_data.shape[1] == 3:
            point_cloud = point_data
        elif point_data.shape[1] == 4:
            I = point_data[:, 3] > intensity_threshold
            point_cloud = point_data[I, :3]
        else:
            raise ValueError("Unsupported format in XYZ file.")
        return point_cloud if not full_data else (point_cloud, point_data)

    return point_cloud if not full_data else (point_cloud,point_data)

def get_cyl_and_plot_data_plotly_ecomodel(point_cloud, title):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],  # 'xy' for 2D, 'scene' for 3D
        subplot_titles=["2D Line Plot", f"Branch {title}"]
    )

    fitter = RobustCylinderFitterEcomodel()
    point_cloud, mean = fitter._normalize_pointcloud(point_cloud)
    fig.add_trace(
        go.Scatter3d(x=point_cloud[:, 0], y=point_cloud[:, 1], z=point_cloud[:, 2], mode='markers', marker=dict(size=3), name='3D Scatter'),
        row=1, col=1
    )

    start, axis,  r, l = fitter.fit(point_cloud)
    circle_projection = fitter.circle_projection
    x = fitter.x
    y = fitter.y
    s = fitter.s

    if circle_projection.any():

        fig.add_trace(
            go.Scatter(x=circle_projection[:, 0], y=circle_projection[:, 1], mode='markers', name='2D Line'),
            row=1, col=2,

        )
        fig.add_shape(
            type="circle",
            xref="x", yref="y",  # Use data coordinates
            x0=x-r, y0=y-r,      # Bottom-left corner of bounding box
            x1=x+r, y1=y+r,      # Top-right corner of bounding box
            line_color="LightSeaGreen",
            fillcolor="PaleTurquoise",
            opacity=0.5
        )
        fig.add_annotation(
            text=f"Diameter: {2*r:0.3f} m<br>S value = {s}",
            xref="x domain", yref="y domain",
            row=1, col=2,
            font={"size": 16},
            showarrow=False,
            x=0, y=1
        )

        fig.update_layout(
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )
        fig.update_layout(
            scene=dict(
                aspectmode='cube'  # Forces x, y, z to be scaled equally
            )
        )
    return fig


def get_cyl_and_plot_data_plotly(point_cloud, title):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],  # 'xy' for 2D, 'scene' for 3D
        subplot_titles=["2D Line Plot", f"Branch {title}"]
    )

    fitter = RobustCylinderFitterEcomodel()
    point_cloud, mean = fitter._normalize_pointcloud(point_cloud)
    fig.add_trace(
        go.Scatter3d(x=point_cloud[:, 0], y=point_cloud[:, 1], z=point_cloud[:, 2], mode='markers', marker=dict(size=3), name='3D Scatter'),
        row=1, col=1
    )

    # Task 1: get cylinder orientation. 
    pc_data = fitter._get_pcs(point_cloud)
    if pc_data is None:
        x = 0
        y = 0
        r = 0
        l = 0
        circle_projection = np.array([])
        # return x, y, r, fig
    else:
        pc1, pc2, pc3, mcd_mean = pc_data

        l = fitter._get_length(point_cloud, pc1)
    
        # Task 2: Robust circle fitting
        circle_params = fitter._fit_circle(point_cloud, pc2, pc3)
        if circle_params is None: 
            print("Failed to Fit Cylinder")
            return fig
        else:
            x, y, r, s = circle_params

            circle_projection = fitter.circle_projection

            # Task 3: Cylinder Start and axis
            center = fitter.get_cylinder_center(pc2, pc3, x, y, mcd_mean)
            axis = fitter.get_cylinder_axis(pc1)
            start = fitter.get_cylinder_start(pc2, pc3, center, axis,  l)


    if circle_projection.any():

        fig.add_trace(
            go.Scatter(x=circle_projection[:, 0], y=circle_projection[:, 1], mode='markers', name='2D Line'),
            row=1, col=2,

        )
        fig.add_shape(
            type="circle",
            xref="x", yref="y",  # Use data coordinates
            x0=x-r, y0=y-r,      # Bottom-left corner of bounding box
            x1=x+r, y1=y+r,      # Top-right corner of bounding box
            line_color="LightSeaGreen",
            fillcolor="PaleTurquoise",
            opacity=0.5
        )
        fig.add_annotation(
            text=f"Diameter: {2*r:0.3f} m<br>S value = {s}",
            xref="x domain", yref="y domain",
            row=1, col=2,
            font={"size": 16},
            showarrow=False,
            x=0, y=1
        )
        # fig.add_annotation(
        #     text=f"S value = {s}",
        #     xref="x domain", yref="y domain",
        #     row=1, col=2,
        #     font={"size": 16},
        #     showarrow=False,
        #     x=0, y=2
        # )
        # fig.add_annotation(text="Absolutely-positioned annotation",
        #           xref="x domain", yref="y domain",
        #           row=1, col=2,
        #           x=0, y=1, showarrow=False)

        fig.update_layout(
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )
        fig.update_layout(
            scene=dict(
                aspectmode='cube'  # Forces x, y, z to be scaled equally
            )
        )




        # ax1 = fig.add_subplot(3, 2, right_pos)
        # ax1.scatter(circle_projection[:, 0], circle_projection[:, 1], s=1)
        # ax1.axis('equal')
        # circle2 = Circle((x, y), r, edgecolor='red', facecolor='none', linewidth=1)
        # ax1.add_patch(circle2)
        # ax1.text(0.01, 0.99, f"Estimated Diameter: {r*2:0.5f} m",
        #                 transform=ax1.transAxes,  # Use axes coordinates
        #                     ha='left', va='top',     # Align text to top-left
        #                     fontsize=12, color='purple')
    return fig

def get_cyl_and_plot_data(point_cloud, num, fig, title = None):
    row = num % 3
    left_pos = row * 2 + 1
    right_pos = row * 2 + 2

    ax2 = fig.add_subplot(3, 2, left_pos, projection='3d')
    ax2.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], color='brown')
    linescale = 10
    # ax2.plot([start[0], axis[0]*linescale], [start[1], axis[1]*linescale], [start[2], axis[2]*linescale], color='red', linewidth=2)
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    ax2.set_zlabel('Z axis')
    ax2.set_title(f'Branch Segment {title}')


    fitter = RobustCylinderFitter()

    # Task 1: get cylinder orientation. 
    point_cloud, mean = fitter._normalize_pointcloud(point_cloud)
    pc_data = fitter._get_pcs(point_cloud)
    if pc_data is None:
        x = 0
        y = 0
        r = 0
        l = 0
        circle_projection = np.array([])
        # return x, y, r, fig
    else:
        pc1, pc2, pc3, mcd_mean = pc_data

        l = fitter._get_length(point_cloud, pc1)
    
        # Task 2: Robust circle fitting
        circle_params = fitter._fit_circle(point_cloud, pc2, pc3)
        if circle_params is None: 
            print("Failed to Fit Cylinder")
            return fig
        else:
            x, y, r, s = circle_params

            circle_projection = fitter.circle_projection

            # Task 3: Cylinder Start and axis
            center = fitter.get_cylinder_center(pc2, pc3, x, y, mcd_mean)
            axis = fitter.get_cylinder_axis(pc1)
            start = fitter.get_cylinder_start(pc2, pc3, center, axis,  l)


    if circle_projection.any():
        ax1 = fig.add_subplot(3, 2, right_pos)
        ax1.scatter(circle_projection[:, 0], circle_projection[:, 1], s=1)
        ax1.axis('equal')
        circle2 = Circle((x, y), r, edgecolor='red', facecolor='none', linewidth=1)
        ax1.add_patch(circle2)
        ax1.text(0.01, 0.99, f"Estimated Diameter: {r*2:0.5f} m",
                        transform=ax1.transAxes,  # Use axes coordinates
                            ha='left', va='top',     # Align text to top-left
                            fontsize=12, color='purple')

    ##### OLD PLOT
    # ax2 = fig.add_subplot(3, 2, right_pos, projection='3d')
    # ax2.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], color='brown')
    # linescale = 10
    # # ax2.plot([start[0], axis[0]*linescale], [start[1], axis[1]*linescale], [start[2], axis[2]*linescale], color='red', linewidth=2)
    # ax2.set_xlabel('X axis')
    # ax2.set_ylabel('Y axis')
    # ax2.set_zlabel('Z axis')
    # ax2.set_title(f'Branch Segment {title}')

    ### New plot
    # ax2 = fig.add_subplot(3, 2, right_pos, projection='3d')
    # ax2.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], color='green', alpha=0.3)
    # ax2.set_xlabel('X axis')
    # ax2.set_ylabel('Y axis')
    # ax2.set_zlabel('Z axis')
    # ax2.set_title(f'Branch Segment {title}')
    # fig.axes
    # linescale = l
    # ax2.scatter(start[0], start[1], start[2],
    #        color='crimson', s=100, label='Start')
    # ax2.plot([start[0], start[0]+ axis[0]*linescale], [start[1],start[1]+ axis[1]*linescale], [start[2], start[2]+axis[2]*linescale], color='red', linewidth=3)


    plt.tight_layout()
    plt.axis('equal')
    # plt.show()

    return fig

def plot_data(point_cloud, start, axis, r, l):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax2 = fig.add_subplot(111, projection='3d')
    ax2.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], color='green', alpha=0.3)
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    ax2.set_zlabel('Z axis')
    ax2.set_title('3D Scatter Plot')
    fig.axes
    linescale = l
    ax2.scatter(start[0], start[1], start[2],
           color='crimson', s=100, label='Start')
    ax2.plot([start[0], start[0]+ axis[0]*linescale], [start[1],start[1]+ axis[1]*linescale], [start[2], start[2]+axis[2]*linescale], color='red', linewidth=3)
    
    # Display
    plt.axis('equal')
    plt.show()



class CylinderPlotter:
    """
    Class which creates matplotlib plots of the segments and cylinders for visualization
    """
    def __init__(self):
        pass

def old_plot():
    folder_name = r"G:\Projects\TreeCanopyLidar\PyTLiDAR\CylinderFitting\segments"
    fitter = RobustCylinderFitting()
    with PdfPages(f"Tree_output.pdf") as pdf:
        fig = plt.figure(figsize=(8, 10))
        fig_in = fig
        for num, branch_file in enumerate(os.listdir(folder_name)):
            
            P = load_point_cloud(f"{folder_name}/{branch_file}")
            segment_getter = SegmentGetter()
            segment_getter.get_more_segments(P)

            for sub_seg_num, sub_segment in enumerate(segment_getter.all_segments):
                print(sub_seg_num)

                segnum_text = branch_file.split("_")[1].split(".")[0] + "_" + str(sub_seg_num)
                np.savetxt(f"output_segments/segment_{segnum_text}.xyz", sub_segment)
                print(f"Segmenting Cylinder {branch_file} {segnum_text}, num points = {len(sub_segment)}")

                if len(sub_segment) < 10:
                    print("SKIPPING")
                    continue

                
                x, y, r, fig_out = get_cyl_and_plot_data(sub_segment, num + sub_seg_num, fig_in, segnum_text)
                fig_in = fig_out
                if (num + sub_seg_num + 1)%3 == 0:
                    pdf.savefig()
                    fig_in = plt.figure(figsize=(8, 10))
                    print("HEELP")




def plot_just_cylinders():
    folder_name = r"G:\Projects\TreeCanopyLidar\PyTLiDAR\CylinderFitting\segments"
    fitter = RobustCylinderFitter()
    with PdfPages(f"Tree_output.pdf") as pdf:
        fig = plt.figure(figsize=(8, 10))
        fig_in = fig
        for num, branch_file in enumerate(os.listdir(folder_name)):
            
            P = load_point_cloud(f"{folder_name}/{branch_file}")
            print(P.shape)
            
            # segment_getter = SegmentGetter()
            # segment_getter.get_more_segments(P)

            # for sub_seg_num, sub_segment in enumerate(segment_getter.all_segments):
            #     print(sub_seg_num)

            #segnum_text = branch_file.split("_")[1].split(".")[0] + "_" + str(sub_seg_num)
            #np.savetxt(f"output_segments/segment_{segnum_text}.xyz", sub_segment)
            #print(f"Segmenting Cylinder {branch_file} {segnum_text}, num points = {len(sub_segment)}")

            # if len(sub_segment) < 10:
            #     print("SKIPPING")
            #     continue

            fig_out = get_cyl_and_plot_data(P, num, fig_in, branch_file)


            fig_in = fig_out
            if (num+ 1)%3 == 0:
                pdf.savefig()
                fig_in = plt.figure(figsize=(8, 10))
                print("HEELP")


def plot_cylinders_plotly():

    folder_name = r"G:\Projects\TreeCanopyLidar\PyTLiDAR\CylinderFitting\segments"
    branches = [899, 100]
    # folder_name = r"xyz_files"
    with open("combined_plots.html", "w") as f:
        for num, branch_file in enumerate(os.listdir(folder_name)):
            branch_num = int(branch_file.split("_")[1].split(".")[0])
            # print(branch_num)
            if branch_num in branches:
                P = load_point_cloud(f"{folder_name}/{branch_file}")
                fig_out = get_cyl_and_plot_data_plotly_ecomodel(P, branch_file)
                fig_out.update_layout(height=600)  # height in pixels

                html = f"""
                <div style="width:50%; margin:auto;">
                    {fig_out.to_html(full_html=(num == 0), include_plotlyjs=('cdn' if num == 0 else False))}
                </div>
                """

                # html = fig_out.to_html(full_html=(num == 0), include_plotlyjs=('cdn' if num == 0 else False))
                print("Writing...")
                f.write(html)
                if num == 10:
                    break


if __name__ == "__main__":
    plot_cylinders_plotly()