#!/usr/bin/env python3
"""
Visualize sentence embeddings in 3D using PCA.
"""

import json
import logging
from pathlib import Path

import click
import numpy as np
import plotly.graph_objects as go
import yaml
from sklearn.decomposition import PCA

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_3d_plot(embeddings_3d, topics, texts, output_file):
    """Create interactive 3D scatter plot."""
    # Create color mapping for topics
    unique_topics = sorted(set(topics))
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    topic_to_color = {topic: colors[i % len(colors)] for i, topic in enumerate(unique_topics)}
    
    # Create traces for each topic
    traces = []
    
    for topic in unique_topics:
        # Filter by topic
        mask = np.array([t == topic for t in topics])
        topic_embeddings = embeddings_3d[mask]
        topic_texts = [texts[i] for i in range(len(texts)) if topics[i] == topic]
        
        # Truncate text for hover
        hover_texts = [t[:100] + '...' if len(t) > 100 else t for t in topic_texts]
        
        trace = go.Scatter3d(
            x=topic_embeddings[:, 0],
            y=topic_embeddings[:, 1],
            z=topic_embeddings[:, 2],
            mode='markers',
            name=topic,
            marker=dict(
                size=6,
                color=topic_to_color[topic],
                opacity=0.8,
                line=dict(width=0.5, color='white')
            ),
            text=hover_texts,
            hovertemplate='<b>%{text}</b><br>' +
                          'Topic: ' + topic + '<br>' +
                          'PC1: %{x:.2f}<br>' +
                          'PC2: %{y:.2f}<br>' +
                          'PC3: %{z:.2f}<br>' +
                          '<extra></extra>'
        )
        
        traces.append(trace)
    
    # Create figure
    fig = go.Figure(data=traces)
    
    fig.update_layout(
        title={
            'text': 'Mechatronics Sentence Embeddings - 3D PCA Visualization',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            xaxis=dict(backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            yaxis=dict(backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            zaxis=dict(backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
        ),
        legend=dict(
            x=1.05,
            y=1,
            font=dict(size=12)
        ),
        width=1200,
        height=800,
        hovermode='closest'
    )
    
    # Save
    fig.write_html(str(output_file))
    logger.info(f"3D plot saved to {output_file}")
    
    # Also save as static image if kaleido is available
    try:
        fig.write_image(str(output_file.with_suffix('.png')), width=1200, height=800)
        logger.info(f"Static image saved to {output_file.with_suffix('.png')}")
    except Exception as e:
        logger.warning(f"Could not save static image: {e}")


@click.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Pipeline config file')
def main(config):
    """Visualize embeddings with 3D PCA."""
    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    eval_dir = Path(config_data['paths']['eval_dir'])
    
    # Load embeddings
    embeddings_file = eval_dir / 'test_embeddings.npz'
    
    if not embeddings_file.exists():
        logger.error(f"Embeddings file not found: {embeddings_file}")
        return
    
    logger.info(f"Loading embeddings from {embeddings_file}")
    data = np.load(embeddings_file, allow_pickle=True)
    
    embeddings = data['embeddings']
    texts = data['texts'].tolist()
    topics = data['topics'].tolist()
    
    logger.info(f"Loaded {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    # Apply PCA
    n_components = config_data['eval']['pca_components']
    logger.info(f"Applying PCA to reduce to {n_components} dimensions...")
    
    pca = PCA(n_components=n_components)
    embeddings_3d = pca.fit_transform(embeddings)
    
    # Print explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print("\n" + "="*60)
    print("PCA ANALYSIS")
    print("="*60)
    print(f"Original dimensions: {embeddings.shape[1]}")
    print(f"Reduced dimensions: {n_components}")
    print(f"\nExplained variance by component:")
    for i, var in enumerate(explained_variance):
        print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")
    print(f"\nCumulative variance: {cumulative_variance[-1]:.4f} ({cumulative_variance[-1]*100:.2f}%)")
    print("="*60)
    
    # Save PCA results
    pca_stats = {
        'n_components': n_components,
        'explained_variance_ratio': explained_variance.tolist(),
        'cumulative_variance': cumulative_variance.tolist()
    }
    
    with open(eval_dir / 'pca_stats.json', 'w') as f:
        json.dump(pca_stats, f, indent=2)
    
    # Save reduced embeddings
    np.savez(
        eval_dir / 'embeddings_3d.npz',
        embeddings_3d=embeddings_3d,
        texts=texts,
        topics=topics
    )
    
    # Create visualization
    logger.info("Creating 3D visualization...")
    output_file = eval_dir / 'pca_3d.html'
    create_3d_plot(embeddings_3d, topics, texts, output_file)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"Interactive 3D plot: {output_file}")
    print(f"Open in browser to explore embeddings!")
    print("="*60)


if __name__ == '__main__':
    main()

