"""Consistent, explicitly registered Times New Roman for every figure artist."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,urllib.request
ROOT=Path(__file__).resolve().parents[3]
FONT_DIR=ROOT/"data/assets/fonts/microsoft_corefonts"
FONT_URL="https://downloads.sourceforge.net/corefonts/times32.exe"
ARCHIVE_SHA256="db56595ec6ef5d3de5c24994f001f03b2a13e37cee27bc25c58f6f43e8f807ab"
FONT_HASHES={'Times.TTF': '4e98adeff8ccc8ef4e3ece8d4547e288ff85fdc9c7ca711a4599c234874bbe86', 'Timesbd.TTF': '4357b63cef20c01661a53c5dae70ffd20cb4765503aaed6d38b17a57c5a90bff', 'Timesbi.TTF': '192e1b0d18e90334e999a99f8c32808d6a2e74b3698b8cd90c943c2249a46549', 'Timesi.TTF': 'c25ae529b4cecdbca148b6ccb862ee0abad770af8b1fd29c8dba619d1b8da78a'}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def setup():
    FONT_DIR.mkdir(parents=True,exist_ok=True)
    archive=FONT_DIR/"times32.exe"
    if not archive.exists():
        with urllib.request.urlopen(FONT_URL,timeout=60) as response:archive.write_bytes(response.read())
    assert sha(archive)==ARCHIVE_SHA256,"Unexpected font archive"
    if not all((FONT_DIR/name).exists() for name in FONT_HASHES):
        extractor=shutil.which("7za") or shutil.which("7z")
        if extractor is None:raise RuntimeError("Extract times32.exe with 7-Zip into "+str(FONT_DIR))
        subprocess.run([extractor,"x","-y","-o"+str(FONT_DIR),str(archive)],check=True)
    for name,expected in FONT_HASHES.items():assert sha(FONT_DIR/name)==expected,name
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),family="Times New Roman",
                source=FONT_URL,archive_sha256=ARCHIVE_SHA256,font_sha256=FONT_HASHES,
                installation="Project-local registration only; no system font installation")
    (FONT_DIR/"manifest.json").write_text(json.dumps(result,indent=2))
    return result
def use_times():
    import matplotlib as mpl
    from matplotlib import font_manager
    for name,expected in FONT_HASHES.items():
        path=FONT_DIR/name
        if not path.exists():raise RuntimeError("Run env/.venv/bin/python repo/src/medwm_eval/figure_style.py first")
        assert sha(path)==expected,name
        font_manager.fontManager.addfont(str(path))
    mpl.rcParams.update({
        "font.family":"Times New Roman","font.serif":["Times New Roman"],
        "font.sans-serif":["Times New Roman"],"font.monospace":["Times New Roman"],
        "pdf.fonttype":42,"ps.fonttype":42,"pdf.use14corefonts":False,
        "mathtext.fontset":"custom","mathtext.rm":"Times New Roman",
        "mathtext.it":"Times New Roman:italic","mathtext.bf":"Times New Roman:bold",
        "mathtext.bfit":"Times New Roman:italic:bold","mathtext.cal":"Times New Roman",
        "mathtext.sf":"Times New Roman","mathtext.tt":"Times New Roman","mathtext.fallback":None,
    })
    for weight,style in [("normal","normal"),("bold","normal"),("normal","italic"),("bold","italic")]:
        actual=font_manager.findfont(font_manager.FontProperties(family="Times New Roman",weight=weight,style=style),fallback_to_default=False)
        assert Path(actual).resolve() in { (FONT_DIR/n).resolve() for n in FONT_HASHES},actual
if __name__=="__main__":print(json.dumps(setup(),indent=2))
