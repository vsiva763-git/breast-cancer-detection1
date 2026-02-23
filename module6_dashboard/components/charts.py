import plotly.graph_objects as go
import plotly.express as px
import json

def render_training_history(history_path='models/training_history.json'):
    with open(history_path) as f:
        history = json.load(f)

    epochs = list(range(1, len(history['train_acc']) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=[a*100 for a in history['train_acc']],
        name='Train Accuracy', mode='lines+markers',
        line=dict(color='#2E86C1', width=2),
        marker=dict(size=5)
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=[a*100 for a in history['val_acc']],
        name='Val Accuracy', mode='lines+markers',
        line=dict(color='#27AE60', width=2),
        marker=dict(size=5)
    ))
    fig.update_layout(
        title='EfficientNet Training History',
        xaxis_title='Epoch',
        yaxis_title='Accuracy (%)',
        height=350,
        legend=dict(x=0.7, y=0.1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240,240,240,0.3)'
    )
    return fig


def render_fl_chart(fl_path='models/fl_metrics.json'):
    with open(fl_path) as f:
        metrics = json.load(f)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=metrics['rounds'],
        y=[a*100 for a in metrics['global_accuracy']],
        name='Global Model', mode='lines+markers',
        line=dict(color='#1E3A5F', width=3),
        marker=dict(size=8, symbol='circle')
    ))
    fig.add_trace(go.Scatter(
        x=metrics['rounds'],
        y=[a*100 for a in metrics['hospital_A_acc']],
        name='Hospital A', mode='lines+markers',
        line=dict(color='#E74C3C', width=1.5, dash='dot'),
        marker=dict(size=5)
    ))
    fig.add_trace(go.Scatter(
        x=metrics['rounds'],
        y=[a*100 for a in metrics['hospital_B_acc']],
        name='Hospital B', mode='lines+markers',
        line=dict(color='#F39C12', width=1.5, dash='dot'),
        marker=dict(size=5)
    ))
    fig.add_trace(go.Scatter(
        x=metrics['rounds'],
        y=[a*100 for a in metrics['hospital_C_acc']],
        name='Hospital C', mode='lines+markers',
        line=dict(color='#27AE60', width=1.5, dash='dot'),
        marker=dict(size=5)
    ))

    fig.add_hrect(
        y0=95, y1=100,
        fillcolor='green', opacity=0.05,
        annotation_text='Target Zone',
        annotation_position='top right'
    )

    fig.update_layout(
        title='Federated Learning — Global Accuracy Across Rounds',
        xaxis_title='Communication Round',
        yaxis_title='Accuracy (%)',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240,240,240,0.3)',
        legend=dict(x=0.75, y=0.1)
    )
    return fig


def render_module_comparison():
    fig = go.Figure(go.Bar(
        x=['Image CNN\n(EfficientNet)', 'Tabular ML\n(XGBoost)', 'NLP\n(TF-IDF LR)', 'Federated\n(FedAvg)'],
        y=[97.74, 96.49, 100.0, 97.70],
        marker_color=['#2E86C1', '#27AE60', '#E67E22', '#8E44AD'],
        text=['97.74%', '96.49%', '100%', '97.70%'],
        textposition='outside'
    ))
    fig.update_layout(
        title='Model Performance Across All Modules',
        yaxis_title='Accuracy (%)',
        yaxis=dict(range=[90, 102]),
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240,240,240,0.3)'
    )
    return fig
