import fitz
import shutil
import subprocess
import tempfile
from pathlib import Path

PDF = Path(__file__).resolve().parent / "Experiment 3.pdf"
WORK = Path(tempfile.mkdtemp(prefix="exp3_latex_"))
BACKUP = WORK / "Experiment 3.before_latex.pdf"
shutil.copy2(PDF, BACKUP)


def formula(name, body, fontsize=10.5):
    tex = rf'''\documentclass[preview,border=0pt]{{standalone}}
\usepackage{{amsmath,amssymb}}
\begin{{document}}
\fontsize{{{fontsize}}}{{{fontsize*1.15}}}\selectfont
{body}
\end{{document}}
'''
    path = WORK / f"{name}.tex"
    path.write_text(tex)
    subprocess.run([
        "pdflatex", "-interaction=nonstopmode", "-halt-on-error",
        "-output-directory", str(WORK), str(path)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return WORK / f"{name}.pdf"


def find_feature_page(doc):
    for i, page in enumerate(doc):
        if "7.7 Feature Maps" in page.get_text():
            return i
    raise RuntimeError("Section 7.7 Feature Maps was not found")


def feature_map_layout_ok(path):
    doc = fitz.open(path)
    page_index = find_feature_page(doc)
    page = doc[page_index]
    rects = []
    for image in page.get_images(full=True):
        rects.extend(page.get_image_rects(image[0]))
    ok = bool(rects)
    for rect in rects:
        ok = ok and rect.x0 >= 40 and rect.x1 <= page.rect.width - 40
        ok = ok and rect.y0 >= 40 and rect.y1 <= page.rect.height - 40
        ok = ok and rect.width <= page.rect.width - 80
    doc.close()
    return ok, page_index


def restore_feature_page(patched_path, original_path, page_index):
    patched = fitz.open(patched_path)
    original = fitz.open(original_path)
    repaired = fitz.open()
    for i in range(patched.page_count):
        source = original if i == page_index else patched
        repaired.insert_pdf(source, from_page=i, to_page=i)
    out = WORK / "Experiment 3.repaired.pdf"
    repaired.save(out, garbage=4, deflate=True, clean=True)
    repaired.close()
    patched.close()
    original.close()
    out.replace(patched_path)


def verify_identity(path):
    doc = fitz.open(path)
    first = doc[0].get_text()
    doc.close()
    if "Experiment 3" not in first:
        raise RuntimeError("Report title is not Experiment 3")
    if "Joshua" not in first:
        raise RuntimeError("Student name is not Joshua")
    if "24110085" not in first:
        raise RuntimeError("Roll number 24110085 is missing")


formulas = {
    "conv": formula("conv", r'\textbf{Convolution:}\quad $Y(i,j)=\displaystyle\sum_m\sum_n X(i+m,j+n)K(m,n)$'),
    "outsize": formula("outsize", r'\textbf{Output size:}\quad $N_{\mathrm{out}}=\left\lfloor\dfrac{N-F+2P}{S}\right\rfloor+1$'),
    "matrices": formula("matrices", r'$X=\left[\begin{smallmatrix}1&2&3\\4&5&6\\7&8&9\end{smallmatrix}\right],\qquad K=\left[\begin{smallmatrix}1&0\\0&1\end{smallmatrix}\right]$'),
    "feature": formula("feature", r'\textbf{Feature map:}\quad $\left[\begin{smallmatrix}6&8\\12&14\end{smallmatrix}\right]$'),
    "poolout": formula("poolout", r'\textbf{Output:}\quad $\left[\begin{smallmatrix}8&3\\6&9\end{smallmatrix}\right]$'),
    "params16": formula("params16", r'$N_{\mathrm{params}}=(3\times3\times3+1)\times16=448$ \text{ trainable parameters}'),
    "ex1": formula("ex1", r'1. $N_{\mathrm{out}}=\left\lfloor\dfrac{64-5+2(2)}{2}\right\rfloor+1=32\;\Rightarrow\;32\times32$'),
    "ex2": formula("ex2", r'2. $N_{\mathrm{params}}=(3\times3\times3+1)\times64=1{,}792$'),
    "relu": formula("relu", r'$\mathrm{ReLU}(x)=\max(0,x)$', 9.5),
    "sigmoid": formula("sigmoid", r'$\sigma(x)=\dfrac{1}{1+e^{-x}}$', 9.5),
}

replacements = [
    (0, fitz.Rect(61.5, 423.8, 505, 442.0), "conv"),
    (0, fitz.Rect(61.5, 444.8, 505, 464.0), "outsize"),
    (2, fitz.Rect(61.5, 101.2, 410, 121.5), "matrices"),
    (2, fitz.Rect(61.5, 230.0, 245, 246.8), "feature"),
    (2, fitz.Rect(61.5, 400.5, 210, 417.3), "poolout"),
    (2, fitz.Rect(61.5, 464.0, 390, 484.0), "params16"),
    (9, fitz.Rect(61.5, 472.5, 520, 491.7), "ex1"),
    (9, fitz.Rect(61.5, 493.7, 520, 513.0), "ex2"),
    (9, fitz.Rect(81.5, 563.2, 202.5, 579.0), "relu"),
    (9, fitz.Rect(81.5, 583.2, 202.5, 599.0), "sigmoid"),
]

max_heights = {
    "conv": 14.5,
    "outsize": 15.5,
    "matrices": 15.0,
    "feature": 12.5,
    "poolout": 12.5,
    "params16": 14.0,
    "ex1": 13.5,
    "ex2": 13.5,
    "relu": 10.5,
    "sigmoid": 10.5,
}

doc = fitz.open(PDF)
for page_number, rect, key in replacements:
    doc[page_number].add_redact_annot(rect, fill=(1, 1, 1))
for page_number in sorted({x[0] for x in replacements}):
    doc[page_number].apply_redactions()

for page_number, rect, key in replacements:
    source = fitz.open(formulas[key])
    source_rect = source[0].rect
    max_height = min(rect.height, max_heights[key])
    scale = min(rect.width / source_rect.width, max_height / source_rect.height)
    width = source_rect.width * scale
    height = source_rect.height * scale
    dest = fitz.Rect(
        rect.x0,
        rect.y0 + (rect.height - height) / 2,
        rect.x0 + width,
        rect.y0 + (rect.height - height) / 2 + height,
    )
    doc[page_number].show_pdf_page(dest, source, 0, keep_proportion=True, overlay=True)
    source.close()

tmp = PDF.with_suffix(".latex.pdf")
doc.save(tmp, garbage=4, deflate=True, clean=True)
doc.close()

ok, feature_page = feature_map_layout_ok(tmp)
if not ok:
    restore_feature_page(tmp, BACKUP, feature_page)

verify_identity(tmp)
ok, _ = feature_map_layout_ok(tmp)
if not ok:
    raise RuntimeError("Section 7.7 feature maps are still outside the page bounds")

tmp.replace(PDF)
