from apollo.data.configs import HistogramConfig
from apollo.data.events import EventCollection
import numpy as np
import plotly.graph_objects as go


def plot_timeline(
        event_collection: EventCollection,
        histogram_config: HistogramConfig,
        draw_records=False,
):
    histogram = event_collection.get_histogram(histogram_config=histogram_config)
    detector = event_collection.detector
    plot_target = np.log10(histogram)
    module_coordinates = detector.module_coordinates

    traces = [
        go.Scatter3d(
            x=module_coordinates[:, 0],
            y=module_coordinates[:, 1],
            z=module_coordinates[:, 2],
            mode="markers",
            marker=dict(
                size=1,
                color="black",  # set color to an array/list of desired values
                colorscale="Viridis",  # choose a colorscale
                opacity=0.1,
            ),
        ),
        go.Scatter3d(
            x=module_coordinates[:, 0],
            y=module_coordinates[:, 1],
            z=module_coordinates[:, 2],
            mode="markers",
            marker=dict(
                size=1,
                color="black",  # set color to an array/list of desired values
                colorscale="Viridis",  # choose a colorscale
                opacity=0.1,
            ),
        ),
    ]

    if draw_records:
        traces.append(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="markers",
                marker=dict(
                    size=3,
                    color="black",  # set color to an array/list of desired values
                    colorscale="Viridis",  # choose a colorscale
                    opacity=0.1,
                ),
            )
        )

    fig = go.Figure(
        data=traces,
    )
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=True,
        height=1000,
        scene=dict(
            xaxis=dict(range=[-1500, 1500], autorange=False),
            yaxis=dict(range=[-1500, 1500], autorange=False),
            zaxis=dict(range=[-1500, 1500], autorange=False),
            aspectmode='cube'
        ),
    )
    fig.update_coloraxes(colorbar_title=dict(text="log10 (det. photons)"))

    sliders = dict(
        active=0,
        yanchor="top",
        xanchor="left",
        currentvalue=dict(
            font=dict(
                size=20,
            ),
            prefix="Time:",
            visible=True,
            xanchor="right",
        ),
        transition=dict(
            duration=300,
            easing="cubic-in-out",
        ),
        pad=dict(
            b=10,
            t=50,
        ),
        len=0.9,
        x=0.1,
        y=0,
        steps=[],
    )

    frames = []
    binned_sources = None
    if draw_records:
        binned_sources = event_collection.get_sources_per_bin(histogram_config=histogram_config)

    # for k in range(5):
    for k in range(histogram.shape[1]):
        plot_target = histogram[:, k]
        mask = (plot_target > 0) & (plot_target != np.nan)
        frame_data = [
            go.Scatter3d(
                x=module_coordinates[mask, 0],
                y=module_coordinates[mask, 1],
                z=module_coordinates[mask, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=plot_target[
                        mask
                    ],  # set color to an array/list of desired values
                    colorscale="Viridis",  # choose a colorscale
                    opacity=0.8,
                    # showscale=True,
                ),
            ),

            go.Scatter3d(
                x=module_coordinates[~mask, 0],
                y=module_coordinates[~mask, 1],
                z=module_coordinates[~mask, 2],
                mode="markers",
                marker=dict(
                    size=1,
                    color="black",  # set color to an array/list of desired values
                    colorscale="Viridis",  # choose a colorscale
                    opacity=0.1,
                ),
            ),
        ]
        if draw_records:
            sources = binned_sources[k]

            source_coordinates = np.array([source.position for source in sources])

            frame_data.append(
                go.Scatter3d(
                    x=source_coordinates[:, 0],
                    y=source_coordinates[:, 1],
                    z=source_coordinates[:, 2],
                    mode="markers",
                    marker=dict(
                        size=3,
                        color="black",
                        colorscale="Viridis",
                        opacity=0.8,
                        # showscale=True,
                    ),
                )
            )
        frames.append(
            go.Frame(
                data=frame_data,
                # traces=[0],
                name=str(k),
            )
        )

        slider_step = dict(
            args=[
                [k],
                dict(
                    frame=dict(
                        duration=300,
                        redraw=True,
                    ),
                    mode="immediate",
                    transition=dict(
                        duration=300,
                    ),
                ),
            ],
            label=k,
            method="animate",
        )
        sliders["steps"].append(slider_step)

    fig.update(frames=frames)

    fig.update_layout(
        sliders=[
            sliders,
        ],
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(
                                    duration=500,
                                    redraw=True,
                                ),
                                fromcurrent=True,
                                transition=dict(
                                    duration=300, easing="quadratic-in-out"
                                ),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(
                                    duration=0,
                                    redraw=True,
                                ),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
                direction="left",
                pad=dict(
                    r=10,
                    t=87,
                ),
                showactive=False,
                x=0.1,
                xanchor="right",
                y=0,
                yanchor="top",
            )
        ],
    )

    fig.show()
    return fig
