"""
AuraGen Generative AI Engine
Handles generative story creation and procedural artwork synthesis.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from transformers import pipeline

class GenAIEngine:
    """Multimodal Generative AI Engine for Text and Visual Concepts."""

    def __init__(self):
        self.text_gen = None
        self._init_text_model()

    def _init_text_model(self):
        try:
            self.text_gen = pipeline("text2text-generation", model="google/flan-t5-small", max_length=512)
        except Exception as e:
            print(f"Text model load warning: {e}")
            self.text_gen = None

    def generate_storyboard(self, concept: str, style: str) -> dict:
        """Generates structured creative lore, scene description, and visual prompt."""
        prompt = f"Write a creative scene story for concept: '{concept}' in style: '{style}'."
        
        if self.text_gen:
            try:
                story_text = self.text_gen(prompt, max_length=200)[0]['generated_text']
            except Exception:
                story_text = f"In a world defined by {concept}, illuminated by {style} atmosphere, futuristic elements merge with ancient lore."
        else:
            story_text = f"In a world defined by {concept}, illuminated by {style} atmosphere, futuristic elements merge with ancient lore."

        return {
            "title": f"{concept.title()} - {style} Chronicles",
            "narrative": story_text,
            "visual_prompt": f"Digital concept art of {concept}, {style} style, dramatic lighting, 8k resolution, photorealistic masterpiece."
        }

    def generate_concept_art(self, prompt: str, style: str, width: int = 512, height: int = 512) -> Image.Image:
        """Generates procedural abstract concept art based on selected aesthetic style."""
        np.random.seed(abs(hash(prompt + style)) % (2**32))
        
        # Color Palettes based on Style
        palettes = {
            "Cyberpunk Neon": [(15, 5, 29), (255, 0, 127), (0, 240, 255), (112, 0, 255)],
            "Fantasy Mystic": [(20, 10, 40), (255, 215, 0), (147, 112, 219), (60, 179, 113)],
            "Sci-Fi Nebula": [(5, 10, 25), (72, 61, 139), (255, 99, 71), (0, 255, 255)],
            "Impressionist Sunset": [(40, 20, 50), (255, 127, 80), (255, 215, 0), (233, 150, 122)],
            "Minimalist Architectural": [(240, 240, 245), (40, 40, 50), (180, 180, 190), (210, 105, 30)]
        }

        bg_color, c1, c2, c3 = palettes.get(style, palettes["Cyberpunk Neon"])
        
        # Base canvas
        img = Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Draw glowing abstract visual geometry & lights
        for _ in range(30):
            shape_type = np.random.choice(['circle', 'line', 'polygon'])
            colors = [c1, c2, c3]
            color = colors[np.random.randint(0, len(colors))]
            
            if shape_type == 'circle':
                x, y = np.random.randint(0, width), np.random.randint(0, height)
                r = np.random.randint(20, 120)
                draw.ellipse([x-r, y-r, x+r, y+r], outline=color, width=np.random.randint(1, 4))
            elif shape_type == 'line':
                x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
                x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
                draw.line([x1, y1, x2, y2], fill=color, width=np.random.randint(2, 6))
            elif shape_type == 'polygon':
                pts = [(np.random.randint(0, width), np.random.randint(0, height)) for _ in range(3)]
                draw.polygon(pts, outline=color)

        # Apply soft atmospheric blur
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        return img
