#!/usr/bin/env python3
"""Convert docs/*.md to PDF using md-to-pdf (npm) with Chrome rendering.
   Also regenerates 答辩PPT.pptx from generate_ppt.py.

   Usage:  python docs/md2pdf.py
   Prereq: npm, Chrome/Chromium installed
"""

import subprocess, os, sys

BASE = os.path.dirname(__file__)
CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

def find_chrome():
    for p in CHROME_PATHS:
        if os.path.exists(p):
            return p
    return None

def main():
    chrome = find_chrome()
    if not chrome:
        print("ERROR: Chrome/Edge not found. Install Chrome or set PUPPETEER_EXECUTABLE_PATH.")
        sys.exit(1)

    env = {**os.environ, "PUPPETEER_EXECUTABLE_PATH": chrome}

    # 1. Regenerate PPT
    print("=" * 60)
    print("Regenerating HDB答辩PPT.pptx...")
    ppt_gen = os.path.join(BASE, "generate_ppt.py")
    if os.path.exists(ppt_gen):
        subprocess.run([sys.executable, ppt_gen], cwd=BASE, check=True)

    # 2. Convert markdown to PDF
    pdf_opts = '{"format":"A4","margin":{"top":"2cm","bottom":"2cm","left":"2.2cm","right":"2.2cm"}}'
    css_file = os.path.join(BASE, "pdf-style.css")

    for md_name in ["项目报告.md", "代码文件说明.md"]:
        md_path = os.path.join(BASE, md_name)
        if not os.path.exists(md_path):
            print(f"SKIP: {md_name} not found")
            continue

        print(f"Converting {md_name} → PDF...")
        cmd = [
            "npx", "md-to-pdf", md_path,
            "--pdf-options", pdf_opts,
            "--stylesheet", css_file,
        ]
        subprocess.run(cmd, cwd=BASE, env=env, check=True)

    print("=" * 60)
    print("All done! Generated files:")
    for f in ["HDB答辩PPT.pptx", "项目报告.pdf", "代码文件说明.pdf"]:
        fp = os.path.join(BASE, f)
        if os.path.exists(fp):
            size_kb = os.path.getsize(fp) / 1024
            print(f"  {f} ({size_kb:.0f} KB)")

if __name__ == "__main__":
    main()
