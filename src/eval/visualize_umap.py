#!/usr/bin/env python3
"""
Visualize sentence embeddings in 3D using UMAP.
UMAP is better at preserving local structure and can exaggerate separations between groups.
"""

import json
import logging
from pathlib import Path

import click
import numpy as np
import plotly.graph_objects as go
import umap
import yaml

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
                          'UMAP1: %{x:.2f}<br>' +
                          'UMAP2: %{y:.2f}<br>' +
                          'UMAP3: %{z:.2f}<br>' +
                          '<extra></extra>'
        )
        
        traces.append(trace)
    
    # Create figure
    fig = go.Figure(data=traces)
    
    fig.update_layout(
        title={
            'text': 'Mechatronics Sentence Embeddings - 3D UMAP Visualization',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        scene=dict(
            xaxis_title='UMAP1',
            yaxis_title='UMAP2',
            zaxis_title='UMAP3',
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
    """Visualize embeddings with 3D UMAP."""
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
    
    # Get UMAP parameters from config
    umap_config = config_data.get('eval', {}).get('umap', {})
    n_components = umap_config.get('n_components', 3)
    n_neighbors = umap_config.get('n_neighbors', 2)  # Lower values emphasize local structure
    min_dist = umap_config.get('min_dist', 0.005)  # Lower values create tighter clusters
    metric = umap_config.get('metric', 'cosine')  # Good for embeddings
    random_state = config_data['eval'].get('random_seed', 42)
    
    logger.info(f"Applying UMAP to reduce to {n_components} dimensions...")
    logger.info(f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
    
    # Apply UMAP
    umap_reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=True
    )
    embeddings_3d = umap_reducer.fit_transform(embeddings)
    
    print("\n" + "="*60)
    print("UMAP ANALYSIS")
    print("="*60)
    print(f"Original dimensions: {embeddings.shape[1]}")
    print(f"Reduced dimensions: {n_components}")
    print(f"Parameters:")
    print(f"  n_neighbors: {n_neighbors}")
    print(f"  min_dist: {min_dist}")
    print(f"  metric: {metric}")
    print("="*60)
    
    # Save UMAP results
    umap_stats = {
        'n_components': n_components,
        'n_neighbors': n_neighbors,
        'min_dist': min_dist,
        'metric': metric,
        'random_state': random_state
    }
    
    with open(eval_dir / 'umap_stats.json', 'w') as f:
        json.dump(umap_stats, f, indent=2)
    
    # Save reduced embeddings
    np.savez(
        eval_dir / 'embeddings_umap_3d.npz',
        embeddings_3d=embeddings_3d,
        texts=texts,
        topics=topics
    )
    
    # Create visualization
    logger.info("Creating 3D visualization...")
    output_file = eval_dir / 'umap_3d.html'
    create_3d_plot(embeddings_3d, topics, texts, output_file)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"Interactive 3D plot: {output_file}")
    print(f"Open in browser to explore embeddings!")
    print("="*60)


if __name__ == '__main__':
    main()

