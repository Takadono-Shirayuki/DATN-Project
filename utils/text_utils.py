"""
Text utilities for drawing Unicode text (Vietnamese) on OpenCV images
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def put_text_vietnamese(img, text, position, font_size=20, color=(0, 255, 0), 
                         font_path=None, thickness=1):
    """
    Vẽ text tiếng Việt lên OpenCV image sử dụng PIL
    
    Args:
        img: OpenCV image (BGR)
        text: Text tiếng Việt
        position: (x, y) - top-left position
        font_size: Kích thước font
        color: Màu BGR tuple (B, G, R)
        font_path: Đường dẫn đến font file (.ttf). Nếu None, dùng font mặc định
        thickness: Độ dày viền (không dùng cho PIL, chỉ để tương thích API)
    
    Returns:
        img: Image with text drawn
    """
    # Convert BGR to RGB for PIL
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Load font
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Try common Vietnamese fonts
            font_paths = [
                'C:/Windows/Fonts/Arial.ttf',
                'C:/Windows/Fonts/arial.ttf',
                'C:/Windows/Fonts/ArialUni.ttf',
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
                '/System/Library/Fonts/Supplemental/Arial.ttf'  # Mac
            ]
            font = None
            for fp in font_paths:
                try:
                    font = ImageFont.truetype(fp, font_size)
                    break
                except:
                    continue
            if font is None:
                font = ImageFont.load_default()
    except Exception as e:
        print(f"Warning: Could not load font, using default: {e}")
        font = ImageFont.load_default()
    
    # Convert BGR to RGB color
    color_rgb = (color[2], color[1], color[0])
    
    # Draw text
    x, y = position
    draw.text((x, y), text, font=font, fill=color_rgb)
    
    # Convert back to BGR for OpenCV
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    return img_bgr


def put_text_with_background(img, text, position, font_size=20, 
                              text_color=(255, 255, 255), bg_color=(0, 0, 0),
                              padding=5, font_path=None):
    """
    Vẽ text tiếng Việt với background để dễ đọc hơn
    
    Args:
        img: OpenCV image (BGR)
        text: Text tiếng Việt
        position: (x, y) - top-left position
        font_size: Kích thước font
        text_color: Màu text BGR tuple
        bg_color: Màu background BGR tuple
        padding: Padding xung quanh text
        font_path: Đường dẫn đến font file
    
    Returns:
        img: Image with text and background drawn
    """
    # Convert BGR to RGB for PIL
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Load font
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font_paths = [
                'C:/Windows/Fonts/Arial.ttf',
                'C:/Windows/Fonts/arial.ttf',
            ]
            font = None
            for fp in font_paths:
                try:
                    font = ImageFont.truetype(fp, font_size)
                    break
                except:
                    continue
            if font is None:
                font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Get text size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Draw background rectangle
    x, y = position
    bg_color_rgb = (bg_color[2], bg_color[1], bg_color[0])
    draw.rectangle(
        [(x - padding, y - padding), 
         (x + text_width + padding, y + text_height + padding)],
        fill=bg_color_rgb
    )
    
    # Draw text
    text_color_rgb = (text_color[2], text_color[1], text_color[0])
    draw.text((x, y), text, font=font, fill=text_color_rgb)
    
    # Convert back to BGR for OpenCV
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    return img_bgr


# Test function
if __name__ == '__main__':
    # Create test image
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)  # Gray background
    
    # Test Vietnamese text
    img = put_text_vietnamese(img, "Người thứ nhất", (50, 50), 
                              font_size=24, color=(0, 255, 0))
    img = put_text_vietnamese(img, "Người thứ 2", (50, 100), 
                              font_size=24, color=(0, 255, 255))
    
    # Test with background
    img = put_text_with_background(img, "Người thứ 3 - Walking", (50, 150),
                                    font_size=24, text_color=(0, 255, 0),
                                    bg_color=(0, 0, 0), padding=5)
    
    # Show result
    cv2.imshow('Vietnamese Text Test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
