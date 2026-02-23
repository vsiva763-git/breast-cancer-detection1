import plotly.graph_objects as go

def render_risk_gauge(score, category, color):
    color_map = {
        'green': '#27AE60',
        'lightgreen': '#82E0AA',
        'orange': '#E67E22',
        'darkorange': '#CA6F1E',
        'red': '#E74C3C'
    }
    gauge_color = color_map.get(color, '#E74C3C')

    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"{category}",
            'font': {'size': 18, 'color': gauge_color}
        },
        number={'suffix': '/100', 'font': {'size': 36}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 2,
                'tickcolor': 'darkgray'
            },
            'bar': {'color': gauge_color, 'thickness': 0.3},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': 'lightgray',
            'steps': [
                {'range': [0, 25],  'color': '#D5F5E3'},
                {'range': [25, 45], 'color': '#ABEBC6'},
                {'range': [45, 60], 'color': '#FDEBD0'},
                {'range': [60, 75], 'color': '#FAD7A0'},
                {'range': [75, 100],'color': '#FADBD8'}
            ],
            'threshold': {
                'line': {'color': gauge_color, 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial'}
    )
    return fig
