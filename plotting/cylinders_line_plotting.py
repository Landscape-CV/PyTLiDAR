import numpy as np
import plotly.graph_objects as go
from plotting.cylinders_plotting import create_cylinder
from plotting.cylinders_plotting import rotation_matrix_from_z


def cylinders_line_plotting(cylinders, scale_factor=1, num_points = 20, file_name = None, overwrite = False, base_fig = None,display = True):
    """
    Plot cylinders as segments with width proportional to radius.

    Parameters:
        cylinders: Contains following properties
            start (list of lists): Starting points of cylinders.
            axis (list of lists): Direction vectors of cylinders.
            radius (list of floats): Radii of cylinders.
        scale_factor (float): Multiplier to convert radius to marker size.

    Returns:
        plotly.graph_objects.Figure
    """
    if base_fig is not None:
        fig = base_fig
    else:
        fig = go.Figure()

    #if colors is None:
    #    colors = ['blue'] * len(starts)
    cylinders = cylinders.copy()  # Avoid modifying the original dictionary
    for key in cylinders:
        cylinders[key] = list(cylinders[key])
    for i in range(np.size(cylinders['radius'])):
        start = np.array(cylinders['start'][i])
        axis = np.array(cylinders['axis'][i])
        radius = cylinders['radius'][i]
        length = cylinders['length'][i]
        color = 'blue'  # Default color

        if True:#radius <= 0.075:
            # Render as line
            end = start + np.array(axis) * length

            # Combine start and end for line plotting
            x_line = [start[0], end[0]]
            y_line = [start[1], end[1]]
            z_line = [start[2], end[2]]

            fig.add_trace(go.Scatter3d(
                x=x_line,
                y=y_line,
                z=z_line,
                mode='lines',
                line=dict(color=color, width=1),
                # line=dict(color=color, width=max(.001,scale_factor * radius * 2)),
                #marker=dict(
                #    size=scale_factor * radius,
                #    color=color,
                #    opacity=0.8
                #),
                name=f'Radius: {radius:.2f}',
                showlegend = False
            ))
        else:
            # Generate cylinder surfaces
            surfaces = create_cylinder(start, axis, radius, length, num_points)

            # Rotate and translate each surface
            rot_matrix = rotation_matrix_from_z(axis / np.linalg.norm(axis))

            for x, y, z in surfaces:
                points = np.stack([x, y, z], axis=-1)
                rotated_points = np.dot(points, rot_matrix.T)
                translated_points = rotated_points + start

                x_t = translated_points[..., 0]
                y_t = translated_points[..., 1]
                z_t = translated_points[..., 2]

                fig.add_trace(go.Surface(
                    x=x_t, y=y_t, z=z_t,
                    colorscale=[[0, color], [1, color]],  # Uniform color
                    showscale=False,
                    opacity=0.8,
                    name=f'Radius: {radius:.2f}',
                ))

    fig.update_layout(
        meta=dict(cylinders=cylinders),
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    # Save the figure to HTML
    if file_name is None:
        html_file = "../cylinder_plot_volume.html"
    elif not overwrite:
        html_file = f"results/cylinder_plot_volume_{file_name}.html"
    else:
        html_file = f"cylinder_plot_volume_.html"
    fig.write_html(html_file, include_plotlyjs=True, full_html=True)

    # Inject JavaScript into the saved HTML
    with open(html_file, "r", encoding='utf-8') as f:
        html = f.read()

    # Inject custom volume panel and script
    injection = """
        <style>
        #volume-panel {
            position: absolute;
            top: 20px;
            left: 20px;
            background: white;
            padding: 10px 15px;
            border: 1px solid #ccc;
            border-radius: 10px;
            font-family: sans-serif;
            box-shadow: 0 0 10px rgba(0,0,0,0.2);
            z-index: 1000;
        }
        </style>
        <div id="volume-panel">Click a point to see nearby volume</div>
        <script>
        document.addEventListener("DOMContentLoaded", function () {
            const plot = document.querySelectorAll('.js-plotly-plot')[0];
            const cylinders = plot.layout.meta.cylinders;
            const panel = document.getElementById("volume-panel");
            const threshold = 0.5; // Distance threshold
        
            plot.on('plotly_click', function (data) {
                const pt = data.points[0];
                const x = pt.x, y = pt.y, z = pt.z;
        
                let totalVolume = 0;
                let ids = [];
        
                const starts = cylinders.start;
                const axis = cylinders.axis;
                const lengths = cylinders.length;
                const radii = cylinders.radius;
        
                for (let i = 0; i < starts.length; i++) {
                    const dx1 = x - starts[i][0];
                    const dy1 = y - starts[i][1];
                    const dz1 = z - starts[i][2];
                    const dist1 = dx1*dx1 + dy1*dy1 + dz1*dz1;
                    const dx2 = x - starts[i][0] - axis[i][0] * lengths[i];
                    const dy2 = y - starts[i][1] - axis[i][1] * lengths[i];
                    const dz2 = z - starts[i][2] - axis[i][2] * lengths[i];
                    const dist2 = dx2*dx2 + dy2*dy2 + dz2*dz2;
                    const dist = Math.min(dist1, dist2);
                    //console.log(dist1, dist2, dist);
                    if (dist < threshold * threshold) {
                        console.log("Cylinder: ", i);
                        const r = radii[i];
                        const h = lengths[i];
                        totalVolume += Math.PI * r * r * h;
                        ids.push(i);
                    }
                }
        
                //panel.innerHTML = `Total Volume Near Click: <b>${totalVolume.toFixed(3)}</b><br>Matched Cylinders: ${ids.join(', ')}`;
                panel.innerHTML = `Total Volume within ${threshold} m of Click Point (${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)}): <br><b>${totalVolume.toFixed(3)}</b> m3`;
        
                // Highlight selected points in red, others blue
                /*let colors = [];
                for (let i = 0; i < starts.length; i++) {
                    colors.push(ids.includes(i) ? 'red' : 'blue');
                }
        
                Plotly.restyle(plot, {'marker.color': [colors]}, [0]);*/
            });
        });
        </script>
    """

    # Inject before closing </body> tag
    html = html.replace("</body>", injection + "\n</body>")

    # Save updated HTML
    with open(html_file, "w", encoding = 'utf-8') as f:
        f.write(html)
    if display:
        print(f"Interactive visualization saved to: {html_file}")

    return fig,html_file


# Example Usage
if __name__ == "__main__":
    cylinders = {
        'start': [
            [0, 0, 0],  # Cylinder 1 start
            [0, 0, 0],  # Cylinder 2 start
            [0, 0, 0],  # Cylinder 3 start
            [2, 2, 2]  # Cylinder 4 start
        ],

        'axis': [
            [1, 0, 0],  # Cylinder 1 axis (along x)
            [0, 1, 0],  # Cylinder 2 axis (along y)
            [0, 0, 1],  # Cylinder 3 axis (along z)
            [0.33, 0.33, 0.33]  # Cylinder 4 axis (diagonal)
        ],

        'radius': [0.2, 0.15, 0.06, 0.05],  # Radii for each cylinder

        'length': [1.0, 0.5, 0.3, 0.6]  # Colors for each cylinder
    }

    fig = cylinders_line_plotting(cylinders, scale_factor=10)