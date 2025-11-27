import gradio as gr
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import umap
from scipy.spatial.distance import pdist, squareform
import io
import threading
import time

# Shared CLIP model (loaded once, used by all tabs)
model = None
processor = None

# Thread-local storage for per-user camera state
thread_local = threading.local()

def load_clip_model():
    """Load CLIP model - shared across all tabs"""
    global model, processor
    if model is None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# ============================================================================
# TAB 1: Single Image Analysis
# ============================================================================

def analyze_image(image, keywords_text):
    """Analyze image with CLIP using comma-separated keywords"""
    if image is None:
        return gr.update(value=None), "Please upload an image"
    
    if not keywords_text or not keywords_text.strip():
        return gr.update(value=None), "Please enter keywords separated by commas"
    
    try:
        model, processor = load_clip_model()
        
        keywords = [k.strip() for k in keywords_text.split(",") if k.strip()]
        if not keywords:
            return gr.update(value=None), "Please provide at least one keyword"
        
        with torch.no_grad():
            inputs = processor(text=keywords, images=image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).squeeze().numpy()
        
        if probs.ndim == 0:
            probs = np.array([probs])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(keywords, probs, color='steelblue', alpha=0.7)
        
        max_idx = np.argmax(probs)
        bars[max_idx].set_color('crimson')
        bars[max_idx].set_alpha(1.0)
        
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{prob:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_xlabel('Keywords', fontsize=12)
        ax.set_title('CLIP Softmax Probabilities', fontsize=14, fontweight='bold')
        ax.set_ylim([0, max(probs) * 1.2])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_image = Image.open(buffer)
        plt.close()
        
        return plot_image, None
        
    except Exception as e:
        return gr.update(value=None), f"Error: {str(e)}"

# ============================================================================
# TAB 2: Multiple Images Comparison
# ============================================================================

def compare_multiple_images(image_files, keyword):
    """Compare multiple images to a keyword using CLIP embeddings"""
    if not image_files or len(image_files) == 0:
        return gr.update(value=None), "Please upload at least one image", gr.update(value=[])
    
    if not keyword or not keyword.strip():
        return gr.update(value=None), "Please enter a keyword", gr.update(value=[])
    
    try:
        model, processor = load_clip_model()
        keyword = keyword.strip()
        
        images = []
        for file_path in image_files:
            try:
                img = Image.open(file_path).convert("RGB")
                images.append(img)
            except Exception as e:
                return gr.update(value=None), f"Error loading image {file_path}: {str(e)}", gr.update(value=[])
        
        if len(images) == 0:
            return gr.update(value=None), "No valid images found", gr.update(value=[])
        
        with torch.no_grad():
            text_emb = model.get_text_features(**processor(text=[keyword], return_tensors="pt", padding=True))
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            
            image_embs = []
            for img in images:
                img_emb = model.get_image_features(**processor(images=img, return_tensors="pt"))
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                image_embs.append(img_emb)
            
            similarities = []
            for img_emb in image_embs:
                sim = (text_emb @ img_emb.T).squeeze().item()
                similarities.append(sim)
        
        similarities = np.array(similarities)
        labels = [f"Image {i+1}" for i in range(len(images))]
        
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_similarities = similarities[sorted_indices]
        sorted_labels = [labels[i] for i in sorted_indices]
        
        closest_idx = np.argmax(similarities)
        closest_image = labels[closest_idx]
        max_sim = similarities[closest_idx]
        min_sim = similarities.min()
        
        def create_thumbnail(img, size=80):
            img_copy = img.copy()
            width, height = img_copy.size
            min_dim = min(width, height)
            left = (width - min_dim) // 2
            top = (height - min_dim) // 2
            img_copy = img_copy.crop((left, top, left + min_dim, top + min_dim))
            img_copy = img_copy.resize((size, size), Image.Resampling.LANCZOS)
            return img_copy
        
        thumbnails = [create_thumbnail(img) for img in images]
        
        fig = plt.figure(figsize=(max(10, len(images) * 1.5), 8))
        ax = fig.add_subplot(111)
        
        bars = ax.bar(labels, similarities, color='steelblue', alpha=0.7)
        bars[closest_idx].set_color('crimson')
        bars[closest_idx].set_alpha(1.0)
        
        for i, (bar, thumb) in enumerate(zip(bars, thumbnails)):
            x_pos = bar.get_x() + bar.get_width() / 2
            y_pos = min_sim - 0.15
            thumb_array = np.array(thumb)
            ax.imshow(thumb_array, extent=[x_pos - 0.3, x_pos + 0.3, y_pos - 0.08, y_pos], 
                     aspect='auto', zorder=3)
            ax.text(x_pos, bar.get_height() + 0.01, f'{similarities[i]:.3f}', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold', zorder=4)
        
        ax.set_ylabel('Similarity Score', fontsize=12)
        ax.set_xlabel('Images', fontsize=12)
        ax.set_title(f'Which image is closest to "{keyword}"?', fontsize=14, fontweight='bold')
        ax.set_ylim([min_sim - 0.25, max_sim + 0.1])
        ax.set_xlim([-0.5, len(images) - 0.5])
        ax.tick_params(axis='x', rotation=45 if len(images) > 5 else 0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        result_text = f"Winner: {closest_image} (similarity: {max_sim:.3f})"
        ax.text(0.5, 0.02, result_text, transform=ax.transAxes, 
               ha='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_image = Image.open(buffer)
        plt.close()
        
        summary = f"Keyword: '{keyword}'\n"
        summary += f"Total images: {len(images)}\n\n"
        summary += "Ranking (highest to lowest similarity):\n"
        summary += "-" * 50 + "\n"
        for rank, (label, sim) in enumerate(zip(sorted_labels, sorted_similarities), 1):
            summary += f"{rank}. {label}: {sim:.4f}\n"
        summary += "\n" + "-" * 50 + "\n"
        summary += f"Winner: {closest_image} with similarity {max_sim:.4f}"
        
        sorted_images = [images[i] for i in sorted_indices]
        return plot_image, summary, sorted_images
        
    except Exception as e:
        return gr.update(value=None), f"Error: {str(e)}", gr.update(value=[])

# ============================================================================
# TAB 3: UMAP Visualization
# ============================================================================

def visualize_umap(image, texts_input):
    """Create UMAP visualization of CLIP embeddings with distance information"""
    if not texts_input or not texts_input.strip():
        return gr.update(value=None), "Please enter at least one text keyword"
    
    try:
        model, processor = load_clip_model()
        
        texts = [t.strip() for t in texts_input.split(",") if t.strip()]
        if not texts:
            return gr.update(value=None), "Please provide at least one keyword"
        
        embeddings = []
        labels = []
        
        with torch.no_grad():
            text_embs = model.get_text_features(**processor(text=texts, return_tensors="pt", padding=True))
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
            text_embs_np = text_embs.cpu().numpy()
            
            for i, text in enumerate(texts):
                embeddings.append(text_embs_np[i])
                labels.append(f"Text: {text}")
        
        if image is not None:
            with torch.no_grad():
                image_emb = model.get_image_features(**processor(images=image, return_tensors="pt"))
                image_emb = image_emb / image_emb.norm()
                image_emb_np = image_emb.cpu().numpy()[0]
                embeddings.append(image_emb_np)
                labels.append("Image")
        
        if len(embeddings) < 2:
            return gr.update(value=None), "Need at least 2 items to visualize distances"
        
        embeddings_array = np.array(embeddings)
        
        n_neighbors = min(5, len(embeddings) - 1) if len(embeddings) > 1 else 1
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
        embeddings_2d = reducer.fit_transform(embeddings_array)
        
        distances = squareform(pdist(embeddings_array, metric='cosine'))
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = ['steelblue'] * len(texts) + (['crimson'] if image is not None else [])
        markers = ['o'] * len(texts) + (['*'] if image is not None else [])
        sizes = [100] * len(texts) + ([300] if image is not None else [])
        
        for i, (emb_2d, label, color, marker, size) in enumerate(zip(embeddings_2d, labels, colors, markers, sizes)):
            ax.scatter(emb_2d[0], emb_2d[1], c=color, marker=marker, s=size, alpha=0.7, edgecolors='black', linewidths=1.5)
            ax.annotate(label, (emb_2d[0], emb_2d[1]), xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        n_items = len(embeddings)
        for i in range(n_items):
            for j in range(i + 1, n_items):
                x1, y1 = embeddings_2d[i]
                x2, y2 = embeddings_2d[j]
                dist = distances[i, j]
                
                ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linewidth=0.5, linestyle='--')
                
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mid_x, mid_y, f'{dist:.3f}', fontsize=7, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
        
        ax.set_xlabel('UMAP Component 1', fontsize=12)
        ax.set_ylabel('UMAP Component 2', fontsize=12)
        ax.set_title('CLIP Embedding Space Visualization (UMAP)\nDistances shown between all pairs', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_image = Image.open(buffer)
        plt.close()
        
        distance_text = "Pairwise Cosine Distances:\n\n"
        for i in range(n_items):
            for j in range(i + 1, n_items):
                dist = distances[i, j]
                distance_text += f"{labels[i]} <-> {labels[j]}: {dist:.4f}\n"
        
        return plot_image, distance_text
        
    except Exception as e:
        return gr.update(value=None), f"Error: {str(e)}"

# ============================================================================
# TAB 4: Live Camera Analysis
# ============================================================================

def get_user_state():
    """Get thread-local state for current user"""
    if not hasattr(thread_local, 'video_capture'):
        thread_local.video_capture = None
        thread_local.is_running = False
        thread_local.keywords = []
    return thread_local

def process_frame(frame, keywords):
    """Process a single frame with CLIP"""
    if not keywords or len(keywords) == 0:
        return frame, None
    
    try:
        model, processor = load_clip_model()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        with torch.no_grad():
            inputs = processor(text=keywords, images=pil_image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).squeeze().numpy()
        
        if probs.ndim == 0:
            probs = np.array([probs])
        
        best_idx = np.argmax(probs)
        best_text = keywords[best_idx]
        best_prob = probs[best_idx]
        
        cv2.putText(frame, f"{best_text}: {best_prob:.3f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(keywords, probs, color='steelblue', alpha=0.7)
        
        bars[best_idx].set_color('crimson')
        bars[best_idx].set_alpha(1.0)
        
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Probability', fontsize=11)
        ax.set_xlabel('Keywords', fontsize=11)
        ax.set_title('CLIP Softmax Probabilities', fontsize=12, fontweight='bold')
        ax.set_ylim([0, max(probs) * 1.2])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_image = Image.open(buffer)
        plt.close()
        
        return frame, plot_image
        
    except Exception as e:
        return frame, None

def start_live_analysis(keywords_text):
    """Start live camera analysis - returns generator for streaming"""
    user_state = get_user_state()
    
    # Always stop any existing session first
    user_state.is_running = False
    if user_state.video_capture:
        try:
            user_state.video_capture.release()
        except:
            pass
        user_state.video_capture = None
    time.sleep(0.2)  # Give time for cleanup
    
    # Parse keywords
    keywords = [k.strip() for k in keywords_text.split(",") if k.strip()]
    if not keywords:
        keywords = ["a person", "a cat", "a dog"]
    
    user_state.keywords = keywords
    
    try:
        user_state.video_capture = cv2.VideoCapture(0)
        if not user_state.video_capture.isOpened():
            yield gr.update(value=None), gr.update(value=None), "Error: Could not open camera (may be in use by another session)"
            return
        
        user_state.is_running = True
        
        while user_state.is_running:
            # Check flag before each operation
            if not user_state.is_running:
                break
                
            ret, frame = user_state.video_capture.read()
            if not ret:
                yield gr.update(value=None), gr.update(value=None), "Error: Could not read frame"
                break
            
            if not user_state.is_running:
                break
            
            annotated_frame, plot_image = process_frame(frame.copy(), keywords)
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            yield rgb_frame, plot_image, "Running"
            
            # Check flag before sleep
            if not user_state.is_running:
                break
            time.sleep(0.2)
            
    except Exception as e:
        yield gr.update(value=None), gr.update(value=None), f"Error: {str(e)}"
    finally:
        # Ensure cleanup
        user_state.is_running = False
        if user_state.video_capture:
            try:
                user_state.video_capture.release()
            except:
                pass
            user_state.video_capture = None

def stop_live_analysis():
    """Stop live camera analysis for current user"""
    user_state = get_user_state()
    
    # Set flag to stop
    user_state.is_running = False
    
    # Release camera
    if user_state.video_capture:
        try:
            user_state.video_capture.release()
        except:
            pass
        user_state.video_capture = None
    
    return gr.update(value=None), gr.update(value=None), "Stopped"

# ============================================================================
# Create Unified Interface with Tabs
# ============================================================================

with gr.Blocks(title="CLIP Analysis Suite") as demo:
    gr.Markdown("# CLIP Analysis Suite")
    gr.Markdown("Unified interface for CLIP analysis. All tabs share the same model for efficient computation.")
    
    with gr.Tabs():
        # Tab 1: Single Image Analysis
        with gr.Tab("Single Image Analysis"):
            gr.Markdown("Upload an image or use camera, then enter comma-separated keywords to see CLIP softmax probabilities.")
            with gr.Row():
                with gr.Column(scale=1):
                    tab1_image = gr.Image(
                        label="Upload Image or Use Camera",
                        type="pil",
                        height=300,
                        sources=["upload", "webcam"]
                    )
                    tab1_keywords = gr.Textbox(
                        label="Keywords (comma-separated)",
                        placeholder="cat, dog, bird, person",
                        lines=2
                    )
                    tab1_btn = gr.Button("Analyze", variant="primary")
                
                with gr.Column(scale=1):
                    tab1_plot = gr.Image(label="Softmax Probabilities", height=400)
                    tab1_error = gr.Textbox(label="Status", interactive=False, visible=False)
            
            tab1_btn.click(
                fn=analyze_image,
                inputs=[tab1_image, tab1_keywords],
                outputs=[tab1_plot, tab1_error]
            )
        
        # Tab 2: Multiple Images Comparison
        with gr.Tab("Multiple Images Comparison"):
            gr.Markdown("Upload multiple images and enter a keyword to see which image is closest to the keyword.")
            with gr.Row():
                with gr.Column(scale=1):
                    tab2_files = gr.File(
                        label="Upload Images",
                        file_count="multiple",
                        file_types=["image"],
                        height=200
                    )
                    tab2_keyword = gr.Textbox(label="Keyword", placeholder="cat", lines=1)
                    tab2_btn = gr.Button("Compare", variant="primary")
                
                with gr.Column(scale=1):
                    tab2_plot = gr.Image(label="Similarity Comparison", height=400)
                    tab2_gallery = gr.Gallery(
                        label="Images (ranked by similarity)",
                        show_label=True,
                        elem_id="gallery",
                        columns=4,
                        rows=2,
                        height="auto"
                    )
                    tab2_result = gr.Textbox(label="Ranking Results", lines=10, interactive=False)
            
            tab2_btn.click(
                fn=compare_multiple_images,
                inputs=[tab2_files, tab2_keyword],
                outputs=[tab2_plot, tab2_result, tab2_gallery]
            )
        
        # Tab 3: UMAP Visualization
        with gr.Tab("UMAP Visualization"):
            gr.Markdown("Upload an optional image and enter comma-separated text keywords to visualize CLIP embeddings in 2D space with UMAP.")
            with gr.Row():
                with gr.Column(scale=1):
                    tab3_image = gr.Image(
                        label="Image (Optional)",
                        type="pil",
                        height=300,
                        sources=["upload", "webcam"]
                    )
                    tab3_texts = gr.Textbox(
                        label="Text Keywords (comma-separated)",
                        placeholder="cat, dog, bird, person, animal",
                        lines=3
                    )
                    tab3_btn = gr.Button("Visualize", variant="primary")
                
                with gr.Column(scale=1):
                    tab3_plot = gr.Image(label="UMAP Visualization", height=500)
                    tab3_distance = gr.Textbox(label="Distance Matrix", lines=15, interactive=False)
            
            tab3_btn.click(
                fn=visualize_umap,
                inputs=[tab3_image, tab3_texts],
                outputs=[tab3_plot, tab3_distance]
            )
        
        # Tab 4: Live Camera Analysis
        with gr.Tab("Live Camera Analysis"):
            gr.Markdown("Enter comma-separated keywords and start live camera analysis with real-time CLIP softmax probabilities.")
            with gr.Row():
                with gr.Column(scale=1):
                    tab4_keywords = gr.Textbox(
                        label="Keywords (comma-separated)",
                        placeholder="a person, a cat, a dog, a phone, a book, a cup",
                        lines=2,
                        value="a person, a cat, a dog, a phone, a book, a cup"
                    )
                    tab4_start = gr.Button("Start Live Analysis", variant="primary")
                    tab4_stop = gr.Button("Stop", variant="stop")
                    tab4_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        value="Enter keywords and click 'Start Live Analysis'"
                    )
                
                with gr.Column(scale=1):
                    tab4_video = gr.Image(label="Live Camera Feed", height=400)
                    tab4_plot = gr.Image(label="Softmax Probabilities", height=300)
            
            tab4_start.click(
                fn=start_live_analysis,
                inputs=[tab4_keywords],
                outputs=[tab4_video, tab4_plot, tab4_status],
                show_progress=False
            )
            
            tab4_stop.click(
                fn=stop_live_analysis,
                inputs=[],
                outputs=[tab4_video, tab4_plot, tab4_status]
            )

if __name__ == "__main__":
    demo.launch(share=True)

