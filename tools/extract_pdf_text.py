from pathlib import Path
import sys
pdf_path = Path(r"C:\Users\ADMIN\Desktop\Github\GR2-Project\Báo_cáo_Học_máy_và_Khai_phá_dữ_liệu.pdf")
out = Path('report_text.txt')
try:
    from PyPDF2 import PdfReader
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'PyPDF2'])
    from PyPDF2 import PdfReader

if not pdf_path.exists():
    print('PDF not found:', pdf_path)
    sys.exit(2)

reader = PdfReader(str(pdf_path))
text_parts = []
for i, page in enumerate(reader.pages):
    try:
        t = page.extract_text() or ''
    except Exception:
        t = ''
    text_parts.append(t)
text = '\n\n'.join(text_parts)
out.write_text(text, encoding='utf-8')
print('WROTE', out)
