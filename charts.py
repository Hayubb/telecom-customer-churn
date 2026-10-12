import plotly.express as px

BLUE = ["#BFDBFE", "#1D4ED8"]
TRANSPARENT = "rgba(0,0,0,0)"


def churn_bar(data, x, height=340, labels=None, hover_data=None):
    fig = px.bar(
        data,
        x=x,
        y="ChurnRate",
        text="ChurnRate",
        color="ChurnRate",
        color_continuous_scale=BLUE,
        labels=labels or {},
        hover_data=hover_data,
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False
    )
    fig.update_layout(
        height=height,
        margin=dict(t=30, b=20, l=10, r=10),
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        coloraxis_showscale=False,
        xaxis_title="",
        yaxis_title="Churn rate (%)",
        yaxis=dict(range=[0, min(100, data["ChurnRate"].max() * 1.25)]),
    )
    return fig
