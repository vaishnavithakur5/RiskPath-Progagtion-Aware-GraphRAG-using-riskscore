import zipfile
import xml.etree.ElementTree as ET
import glob
import os

files = glob.glob(r'C:\Users\ausu\Downloads\RiskPath*.pptx') + glob.glob(r'C:\Users\ausu\Downloads\RiskPath*.docx')
print(f"Found {len(files)} files to inspect.")

search_terms = ['inherited', 'propagation', 'coefficient', 'total risk', 'own risk', 'pwrs', 'causal', 'alpha', 'formula', 'reliability']

for f in files:
    print(f"\n==================================================")
    print(f"FILE: {os.path.basename(f)}")
    print(f"==================================================")
    try:
        with zipfile.ZipFile(f, 'r') as z:
            for name in z.namelist():
                if name.startswith('ppt/slides/slide') or name == 'word/document.xml':
                    root = ET.fromstring(z.read(name))
                    text = ' '.join(root.itertext())
                    text_lower = text.lower()
                    found = [t for t in search_terms if t in text_lower]
                    if any(t in text_lower for t in ['inherited', 'own risk', 'coefficient', 'pwrs', 'total risk']):
                        print(f"  -> Found in {name}: {found}")
                        # print surrounding sentences
                        sentences = text.replace('.', '.\n').splitlines()
                        for s in sentences:
                            if any(k in s.lower() for k in ['inherited', 'own risk', 'coefficient', 'total risk', 'pwrs', 'formula', 'alpha', 'beta', 'risk(']):
                                print(f"     * {s.strip()[:140]}")
    except Exception as e:
        print(f"  Error: {e}")
