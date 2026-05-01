import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA

# Add Code/src to path to import babelbias
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Code", "src"))

from babelbias.paths import PROCESSED_LEADS_DIR, PROJECT_ROOT
from babelbias.debias import load_control_embeddings, language_subspace_basis, project_out

OUT_PATH = PROJECT_ROOT / "assets" / "pca_space.gif"

# Filtered languages
LANGS = ("ru", "uk")
COLORS = {
    "ru": "#d62728",  # Red
    "uk": "#1f77b4",  # Blue
}

# Target topic for the static zoom
TARGET_TOPIC = "2014 Russian annexation of Crimea"

def load_data():
    embs, meta = [], []
    for fn in os.listdir(PROCESSED_LEADS_DIR):
        if not fn.endswith(".json"):
            continue
        with open(PROCESSED_LEADS_DIR / fn) as f:
            d = json.load(f)
        
        if d.get("type") != "conflict" or d.get("language") not in LANGS:
            continue
            
        embs.append(d["embedding"])
        meta.append({
            "topic": d.get("conflict") or fn.rsplit("_", 1)[0],
            "language": d["language"],
        })
    return np.array(embs), pd.DataFrame(meta)

def main():
    print("Loading data...")
    X_raw, df = load_data()
    
    print("Loading controls for debiasing...")
    ctrl_X, ctrl_langs = load_control_embeddings(PROCESSED_LEADS_DIR, LANGS)
    basis = language_subspace_basis(ctrl_X, ctrl_langs, LANGS)
    X_db = project_out(X_raw, basis)
    
    print(f"Loaded {len(X_raw)} embeddings. Performing PCA...")
    pca_raw = PCA(n_components=3)
    coords_raw = pca_raw.fit_transform(X_raw)
    
    pca_db = PCA(n_components=3)
    coords_db = pca_db.fit_transform(X_db)
    
    topic_counts = df["topic"].value_counts()
    full_topics = topic_counts[topic_counts == 2].index
    
    fig = plt.figure(figsize=(6, 5), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    
    def setup_ax(ax):
        ax.set_facecolor("white")
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    # Animation parameters
    n_spin = 90
    n_zoom = 60
    state_frames = n_spin + n_zoom
    total_frames = state_frames * 2
    
    # Global limits for consistent framing
    all_coords = np.vstack([coords_raw, coords_db])
    global_center = all_coords.mean(axis=0)
    global_max_half_range = (all_coords.max(axis=0) - all_coords.min(axis=0)) * 0.5

    def update(frame):
        ax.clear()
        setup_ax(ax)
        
        cycle = frame // state_frames
        is_debiased = cycle % 2 == 1
        f = frame % state_frames
        
        coords = coords_db if is_debiased else coords_raw
        
        # Determine target center for zoom
        target_indices = df[df["topic"] == TARGET_TOPIC].index
        if len(target_indices) == 2:
            target_pts = coords[target_indices]
            target_center = target_pts.mean(axis=0)
        else:
            target_center = global_center

        # Phase logic
        if f < n_spin:
            # Phase 1: Rotating
            angle = (frame * 1.5) % 360
            zoom_factor = 1.1 - 0.3 * (f / n_spin) # Gradual zoom in
            current_center = global_center
            current_range = global_max_half_range * zoom_factor
            show_labels = False
            title_prefix = "PCA (RU vs UK)"
        else:
            # Phase 2: Static Zoom
            angle = ( (cycle * state_frames + n_spin) * 1.5 ) % 360 # Fixed at last spin angle
            zoom_progress = (f - n_spin) / n_zoom
            zoom_factor = 0.8 - 0.6 * zoom_progress # Deep zoom
            
            # Smoothly transition center from global to target
            current_center = global_center + (target_center - global_center) * zoom_progress
            current_range = global_max_half_range * zoom_factor
            show_labels = True
            title_prefix = f"ZOOM: {TARGET_TOPIC}"

        ax.view_init(elev=20, azim=angle)
        
        state_text = "DEBIASED" if is_debiased else "RAW"
        ax.set_title(f"{title_prefix}\n{state_text} EMBEDDING SPACE", fontsize=11, pad=8)
        
        # Plot all connections faintly
        for topic in full_topics:
            idx = df[df["topic"] == topic].index
            if len(idx) == 2:
                pts = coords[idx]
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="gray", alpha=0.1, linewidth=0.5)
        
        # Highlight target connection
        if len(target_indices) == 2:
            pts = coords[target_indices]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="black", alpha=0.6, linewidth=1.2)

        # Plot points
        for lang in LANGS:
            mask = df["language"] == lang
            ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2], 
                       c=COLORS[lang], label=lang.upper(), s=15, alpha=0.6, edgecolors="white", linewidth=0.2)
        
        # Labels for target event
        if show_labels and len(target_indices) == 2:
            for idx in target_indices:
                pt = coords[idx]
                lang = df.loc[idx, "language"].upper()
                ax.text(pt[0], pt[1], pt[2], f"{lang}: {TARGET_TOPIC}", 
                        fontsize=7, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        ax.legend(loc="upper right", frameon=False, fontsize=8)
        
        # Apply camera limits
        ax.set_xlim(current_center[0] - current_range[0], current_center[0] + current_range[0])
        ax.set_ylim(current_center[1] - current_range[1], current_center[1] + current_range[1])
        ax.set_zlim(current_center[2] - current_range[2], current_center[2] + current_range[2])
        
        return fig,

    print(f"Creating animation ({total_frames} frames)...")
    ani = FuncAnimation(fig, update, frames=total_frames, interval=50)
    
    print(f"Saving to {OUT_PATH}...")
    writer = PillowWriter(fps=20)
    ani.save(OUT_PATH, writer=writer)
    plt.close()
    print("Done!")

if __name__ == "__main__":
    main()
