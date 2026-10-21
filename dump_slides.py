import zipfile
import xml.etree.ElementTree as ET
import sys

# Ensure stdout uses utf-8
sys.stdout.reconfigure(encoding='utf-8')

pptx_path = r'C:\Users\ausu\Downloads\RiskPath_Review2_PPT.pptx'
print(f"Reading {pptx_path}...")

with zipfile.ZipFile(pptx_path, 'r') as z:
    slides = sorted([n for n in z.namelist() if n.startswith('ppt/slides/slide') and n.endswith('.xml')],
                    key=lambda x: int(''.join(filter(str.isdigit, x))))
    for slide_name in slides:
        print(f"\n==========================================")
        print(f"SLIDE: {slide_name}")
        print(f"==========================================")
        root = ET.fromstring(z.read(slide_name))
        for p in root.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}p'):
            line = ''.join(p.itertext()).strip()
            if line:
                print(line)
