from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go

from apollo.data.configs import HistogramConfig
from apollo.data.events import EventCollection


def plot_timeline(
    event_collection: EventCollection,
    histogram_config: HistogramConfig,
    draw_sources: bool = False,
    show: bool = True,
) -> go.Figure:
    """Create plots for a event collection based on their histograms.

    Args:
        event_collection: Collection of events to include in timeline
        histogram_config: HistogramConfig to create the events for
        draw_sources: Include sources in plot
        show: whether to show or not at the end

    Returns:

    """
    histogram = event_collection.get_histogram(histogram_config=histogram_config)
    detector = event_collection.detector
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

    if draw_sources:
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

    figure = go.Figure(data=traces)
    figure.update_layout(
        showlegend=False,
        coloraxis_showscale=True,
        height=1000,
        scene=dict(
            xaxis=dict(range=[-1500, 1500], autorange=False),
            yaxis=dict(range=[-1500, 1500], autorange=False),
            zaxis=dict(range=[-1500, 1500], autorange=False),
            aspectmode="cube",
        ),
    )
    figure.update_coloraxes(colorbar_title=dict(text="log10 (det. photons)"))

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
    )  # type: Dict[str, Any]

    frames = []
    binned_sources = []
    if draw_sources:
        binned_sources = event_collection.get_sources_per_bin(
            histogram_config=histogram_config
        )

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
        if draw_sources:
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
        steps = sliders["steps"]
        steps.append(slider_step)

    figure.update(frames=frames)

    figure.update_layout(
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

    if show:
        figure.show()

    return figure


def plot_histogram(histogram: np.typing.NDArray[np.single]) -> go.Heatmap:
    """Plots histogram based on events.

    Args:
        histogram: Histogram to plot

    Returns:

    """
    result = np.where(histogram > 0.0000000001, histogram, 0)

    heatmap = go.Heatmap(z=np.log(result, out=result, where=result > 0), colorbar=None)
    return heatmap
