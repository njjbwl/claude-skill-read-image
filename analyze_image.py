#!/usr/bin/env python3
"""Image analysis for Claude Code. --ocr-lang, --vision, --face, --video"""
import json,sys,os,io,base64,argparse,subprocess,tempfile,shutil
from pathlib import Path
import numpy as np

class _NumpyEncoder(json.JSONEncoder):
    def default(self,obj):
        try:
            if isinstance(obj,(np.integer,)): return int(obj)
            if isinstance(obj,(np.floating,)): return float(obj)
            if isinstance(obj,(np.ndarray,)): return obj.tolist()
        except: pass
        return super().default(obj)

DEP={"pillow":False,"numpy":False,"pytesseract":False,"tesseract_bin":False,"requests":False,"sklearn":False,"vision_api_key":False,"ffmpeg":False}
_FFMPEG_BIN=None
for _c in [r"D:\claude\LivePortrait\ffmpeg\ffmpeg.exe",shutil.which("ffmpeg")]:
    if _c and os.path.exists(str(_c)): _FFMPEG_BIN=str(_c); DEP["ffmpeg"]=True; break
try: from PIL import Image,ExifTags; DEP["pillow"]=True
except: Image=ExifTags=None
try: import numpy as np; DEP["numpy"]=True
except: np=None
try:
    import pytesseract as _pt; DEP["pytesseract"]=True
    for _c in [r"C:\Program Files\Tesseract-OCR\tesseract.exe",r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe","/usr/bin/tesseract","/usr/local/bin/tesseract"]:
        if os.path.exists(_c): _pt.pytesseract.tesseract_cmd=_c; DEP["tesseract_bin"]=True; break
    if not DEP["tesseract_bin"]: DEP["tesseract_bin"]=bool(shutil.which("tesseract"))
except: _pt=None
try: import requests as _req; DEP["requests"]=True
except: _req=None
try: from sklearn.cluster import KMeans as _KM; DEP["sklearn"]=True
except: _KM=None
SILICONFLOW_API_KEY=os.environ.get("SILICONFLOW_API_KEY","") or ""
DEP["vision_api_key"]=bool(SILICONFLOW_API_KEY)
try:
    import insightface
    from insightface.app import FaceAnalysis as _IFaceAnalysis; DEP["insightface"]=True
    _mr=os.path.expanduser("~/.insightface")
    if os.path.isdir(os.path.join("D:/claude/LivePortrait/pretrained_weights/insightface","models","buffalo_l")): _mr="D:/claude/LivePortrait/pretrained_weights/insightface"
    try:
        _face_app=_IFaceAnalysis(name='buffalo_l',root=_mr,providers=['CPUExecutionProvider'],allowed_modules=['detection','landmark_2d_106','genderage','recognition'])
        _face_app.prepare(ctx_id=0,det_size=(640,640)); DEP["face_model_loaded"]=True
    except: _face_app=None; DEP["face_model_loaded"]=False
except: _face_app=None; DEP["insightface"]=False; DEP["face_model_loaded"]=False

def _to_rgb(img):
    if img.mode=="RGBA": bg=Image.new("RGB",img.size,(255,255,255)); bg.paste(img,mask=img.split()[3]); return bg
    return img if img.mode=="RGB" else img.convert("RGB")
def _downscale(img,m=2000):
    if max(img.width,img.height)<=m: return img,False
    r=m/max(img.width,img.height); return img.resize((int(img.width*r),int(img.height*r)),Image.LANCZOS),True
def extract_metadata(img,path):
    md={"format":str(img.format or "unknown"),"mode":img.mode,"width":img.width,"height":img.height,"aspect_ratio":round(img.width/img.height,3),"file_size_bytes":path.stat().st_size,"file_size_kb":round(path.stat().st_size/1024,1)}
    if hasattr(img,"info") and img.info.get("dpi"): md["dpi"]=img.info["dpi"]
    return md
def extract_exif(img):
    if not hasattr(img,"_getexif") or not img._getexif(): return None
    exif={}
    for tid,v in img._getexif().items():
        t=ExifTags.TAGS.get(tid,tid)
        if isinstance(v,(str,int,float)): exif[t]=v
        elif isinstance(v,bytes):
            try: exif[t]=v.decode("utf-8",errors="replace")[:200]
            except: pass
    return exif if exif else None
def analyze_colors(img):
    t=img.copy()
    if max(t.width,t.height)>200: r=200/max(t.width,t.height); t=t.resize((int(t.width*r),int(t.height*r)),Image.LANCZOS)
    if not DEP["sklearn"] or not DEP["numpy"]: return None
    a=np.array(t).reshape(-1,3); km=_KM(n_clusters=5,n_init=3,random_state=42,max_iter=300).fit(a)
    cts=np.bincount(km.labels_,minlength=5); dom=[]
    for i in np.argsort(-cts):
        c=km.cluster_centers_[i].astype(int); pct=round(cts[i]/len(km.labels_)*100,1) if len(km.labels_) else 0
        dom.append({"rgb":f"rgb({c[0]},{c[1]},{c[2]})","hex":f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}","pixel_count":int(cts[i]),"percentage":pct})
    return {"dominant":dom,"method":"kmeans"}
def analyze_luminance(img):
    if not DEP["numpy"]: return None
    a=np.array(img.convert("L"),dtype=np.float32)
    return {"mean":round(float(a.mean()),1),"std":round(float(a.std()),1),"min":int(a.min()),"max":int(a.max()),"dark_pct":round(float((a<64).sum()/a.size*100),1),"bright_pct":round(float((a>192).sum()/a.size*100),1)}
def _hb(c):
    h=c.lstrip("#"); return int(h[0:2],16)*0.299+int(h[2:4],16)*0.587+int(h[4:6],16)*0.114
def heuristics(md,lum,colors=None,ocr=None):
    h,sig={},{}
    if lum: sig.update({"dark":lum["mean"]<60,"bright":lum["mean"]>180,"high_contrast":lum["std"]>40,"bimodal":lum["dark_pct"]>30 and (lum["max"]-lum["mean"])>60})
    if colors and colors.get("dominant"):
        d=colors["dominant"]; tp=d[0]["percentage"] if d else 0
        sig.update({"single_dominant":tp>80,"colorful":len(d)>=4 and tp<50})
        dc=sum(1 for c in d[:3] if _hb(c["hex"])<80); bc=sum(1 for c in d[:3] if _hb(c["hex"])>180)
        sig.update({"mostly_dark":dc>=2,"mostly_bright":bc>=2})
    if ocr and "error" not in ocr: sig["has_text"]=ocr.get("lines",0)>0
    else: sig["has_text"]=False
    if md:
        mp=(md.get("width",0)*md.get("height",0))/1e6; ar=md.get("aspect_ratio",0)
        sig.update({"large":mp>5,"widescreen":ar and abs(ar-16/9)<0.15,"portrait":ar and 0<ar<0.9})
    h.update({"terminal_screenshot":sig.get("dark") and sig.get("bimodal") and sig.get("has_text") and not sig.get("large"),"photo":sig.get("large") or (sig.get("colorful") and not sig.get("has_text")),"document_scan":sig.get("bright") and sig.get("has_text") and sig.get("single_dominant") and not sig.get("colorful")})
    if sig.get("widescreen"): h["widescreen"]=True
    if sig.get("portrait") and md.get("width",0)<md.get("height",0): h["portrait"]=True
    return h if any(v for v in h.values()) else None
def _ocr_preprocess(img):
    if not DEP["numpy"]: return img
    a=np.array(img.convert("L"),dtype=np.uint8); mb=float(a.mean())
    if mb<80: return img
    th=max(mb*0.85,128) if mb>240 else max(mb*0.8,100)
    b=(np.array(img.convert("L"),dtype=np.uint8)>th).astype(np.uint8)*255
    r=Image.fromarray(b).convert("RGB")
    if min(r.size)<500:
        s=max(1.0,800/min(r.size))
        if s>1.2: r=r.resize((int(r.width*s),int(r.height*s)),Image.LANCZOS)
    return r
def run_ocr(img,lang="eng"):
    if not DEP["pytesseract"]: return {"error":"pytesseract not installed"}
    if not DEP["tesseract_bin"]: return {"error":"Tesseract not found"}
    try:
        try: avail=_pt.get_languages(config="")
        except: avail=["eng"]
        req=lang.replace("+"," ").split(); miss=[l for l in req if l not in avail]
        lu=lang if len(miss)!=len(req) else ([l for l in avail if l!="osd"][0] if [l for l in avail if l!="osd"] else "eng")
        p=_ocr_preprocess(img)
        try:
            osd=_pt.image_to_osd(p,output_type=_pt.Output.DICT)
            if abs(float(osd.get("rotate",0)))>0.5: p=p.rotate(-float(osd.get("rotate",0)),expand=True,fillcolor=(255,255,255))
        except: pass
        d=_pt.image_to_data(p,lang=lu,output_type=_pt.Output.DICT)
        t=_pt.image_to_string(p,lang=lu)
        tl=[l.strip() for l in t.split("\n") if l.strip()]; words=[]
        for i in range(len(d["text"])):
            w=d["text"][i].strip(); c=int(d["conf"][i]) if d["conf"][i]!="-1" else 0
            if w and c>0: words.append({"text":w,"confidence":c,"bbox":{"x":d["left"][i],"y":d["top"][i],"w":d["width"][i],"h":d["height"][i]}})
        r={"language_used":lu,"language_requested":lang,"lines":len(tl),"words":len(words),"total_chars":len(t),"text":t[:3000],"truncated":len(t)>3000,"word_details":words[:200]}
        if words: r["avg_confidence"]=round(sum(w["confidence"] for w in words)/len(words),1)
        return r
    except Exception as e: return {"error":str(e)}
VISION_MODEL="Qwen/Qwen3-VL-32B-Instruct"
def call_vision(path,prompt=None):
    if not DEP["requests"]: return {"error":"requests not installed"}
    if not DEP["vision_api_key"]: return {"error":"SILICONFLOW_API_KEY not set"}
    if not prompt: prompt="Describe: scene, objects, text, colors, image type"
    try:
        img=Image.open(path); fmt=img.format or "PNG"; buf=io.BytesIO()
        sf=fmt if fmt.upper() in ("PNG","JPEG","JPG","WEBP") else "PNG"
        img.save(buf,format=sf)
        b64=base64.b64encode(buf.getvalue()).decode(); ext=sf.lower().replace("jpeg","jpg")
        r=_req.post("https://api.siliconflow.cn/v1/chat/completions",headers={"Authorization":f"Bearer {SILICONFLOW_API_KEY}","Content-Type":"application/json"},json={"model":VISION_MODEL,"messages":[{"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":f"data:image/{ext};base64,{b64}"}}]}],"max_tokens":1024,"temperature":0.3},timeout=60)
        r.raise_for_status(); data=r.json()
        c=data.get("choices",[{}])[0].get("message",{}).get("content","")
        if not c: return {"error":"Empty response"}
        return {"description":c[:2000],"model":VISION_MODEL,"usage":data.get("usage",{})}
    except Exception as e: return {"error":f"Vision API: {e}"}
def run_face_detection(path):
    if not DEP.get("insightface"): return {"error":"insightface not installed"}
    if not DEP.get("face_model_loaded"): return {"error":"Face model not loaded"}
    try:
        import cv2
        img=cv2.imread(path)
        if img is None: img=np.array(Image.open(path).convert("RGB"))[:,:,::-1]
        faces=_face_app.get(img); r={"count":len(faces),"faces":[]}
        for i,f in enumerate(faces):
            info={"face_index":i+1,"bbox":[round(float(x),2) for x in f.bbox],"confidence":round(float(f.det_score),3)}
            if hasattr(f,"age") and f.age is not None: info["age"]=int(f.age)
            if hasattr(f,"gender") and f.gender is not None: info["gender"]="male" if int(f.gender)==1 else "female"
            if hasattr(f,"kps") and f.kps is not None: info["keypoints_5"]=[[round(float(x),2) for x in pt] for pt in f.kps]
            if hasattr(f,"landmark_2d_106") and f.landmark_2d_106 is not None:
                lm=f.landmark_2d_106; xs=[p[0] for p in lm]; ys=[p[1] for p in lm]
                if max(xs)-min(xs)>0: info["face_aspect_ratio"]=round((max(ys)-min(ys))/(max(xs)-min(xs)),2)
                info["landmarks_106"]=[[round(float(x),2) for x in pt] for pt in lm]
            if hasattr(f,"embedding") and f.embedding is not None: info["embedding_available"]=True
            if f.bbox is not None:
                _tmp=Image.open(path)
                fa=(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])
                if _tmp.width*_tmp.height>0: info["face_area_ratio"]=round(float(fa)/float(_tmp.width*_tmp.height),3)
            r["faces"].append(info)
        return r
    except Exception as e: return {"error":f"Face detection: {e}"}

_VIDEO_EXT={'.mp4','.avi','.mov','.mkv','.webm','.flv','.wmv'}
def analyze_video(path,fps=1.0,enable_face=True):
    if not DEP.get("ffmpeg"): return {"error":"ffmpeg not found"}
    if not os.path.exists(path): return {"error":f"Video not found: {path}"}
    r={"file":path,"total_frames":0,"analyzed_frames":0,"frames_with_faces":0,"face_timeline":[]}
    td=tempfile.mkdtemp(prefix="vid_")
    try:
        try:
            p=subprocess.run([_FFMPEG_BIN,"-i",path],capture_output=True,text=True,timeout=30,stderr=subprocess.PIPE)
            for ln in p.stderr.split("\n"):
                if "Duration" in ln and ":" in ln:
                    ds=ln.split("Duration:")[1].split(",")[0].strip() if "Duration:" in ln else ""
                    if ds: ps=ds.split(":"); r["duration_seconds"]=float(ps[0])*3600+float(ps[1])*60+float(ps[2])
        except: pass
        op=os.path.join(td,"frame_%04d.jpg")
        subprocess.run([_FFMPEG_BIN,"-i",path,"-vf",f"fps={fps}","-q:v","2","-y",op],capture_output=True,timeout=600)
        fs=sorted([f for f in os.listdir(td) if f.endswith(".jpg")])
        r["total_frames"]=len(fs); r["analyzed_frames"]=len(fs)
        if not enable_face or not DEP.get("insightface") or not DEP.get("face_model_loaded"): return r
        tl,fc,fw,ap=[],0,0,{}
        for i,fn in enumerate(fs):
            fp=os.path.join(td,fn)
            try:
                fr=run_face_detection(fp)
                if "error" not in fr and fr.get("count",0)>0:
                    fc+=fr["count"]; fw+=1
                    entry={"frame_index":i,"timestamp":round(i/fps,2),"faces":fr["count"]}
                    if fr.get("faces"):
                        entry["face_details"]=[{"bbox":x["bbox"],"confidence":x.get("confidence")} for x in fr["faces"]]
                    tl.append(entry)
            except: pass
        r["frames_with_faces"]=fw; r["total_face_detections"]=fc; r["face_timeline"]=tl
        if fs: r["face_coverage_pct"]=round(fw/len(fs)*100,1)
        return r
    except subprocess.TimeoutExpired: return {"error":"Video processing timed out"}
    except Exception as e: return {"error":f"Video failed: {e}"}
    finally:
        try: shutil.rmtree(td)
        except: pass

def analyze_image(path,ocr_lang="eng",enable_vision=False,enable_face=False):
    p=Path(path); r={"file":str(p.resolve()),"ocr_lang":ocr_lang,"dependencies":{k:v for k,v in DEP.items() if k!="face_app"}}
    if not p.exists(): return {"error":f"Not found: {path}"}
    if not DEP["pillow"]: return {"error":"Pillow not installed","file":str(p.resolve())}
    try: img=Image.open(p)
    except Exception as e: return {"error":f"Cannot open: {e}","file":str(p.resolve())}
    r["metadata"]=extract_metadata(img,p)
    ex=extract_exif(img)
    if ex: r["exif"]=ex
    rgb_img,wd=_downscale(_to_rgb(img),2000)
    if wd: r["downscaled"]={"original_size":f"{img.width}x{img.height}","new_size":f"{rgb_img.width}x{rgb_img.height}"}
    co=analyze_colors(rgb_img)
    if co: r["colors"]=co
    lu=analyze_luminance(rgb_img)
    if lu: r["luminance"]=lu
    oc=run_ocr(rgb_img,ocr_lang)
    r["ocr"]=oc
    h=heuristics(r.get("metadata",{}),lu,colors=r.get("colors"),ocr=oc)
    if h: r["heuristics"]=h
    if enable_vision:
        if DEP["vision_api_key"] and DEP["requests"]: r["vision"]=call_vision(path)
        else: r["vision"]={"note":"Vision API disabled"}
    if enable_face: r["face"]=run_face_detection(str(p.resolve()))
    return r

def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8","utf8"):
        try: sys.stdout.reconfigure(encoding="utf-8",errors="replace")
        except: pass
    ps=argparse.ArgumentParser(description="Image/video analysis for Claude Code")
    ps.add_argument("image",nargs="?",help="Path")
    ps.add_argument("--ocr-lang",default="eng",help="OCR language")
    ps.add_argument("--vision",action="store_true",help="Vision API")
    ps.add_argument("--face",action="store_true",help="Face detection")
    ps.add_argument("--video",action="store_true",help="Video analysis")
    ps.add_argument("--video-fps",type=float,default=1.0,help="FPS for video (default: 1)")
    ps.add_argument("--check",action="store_true",help="Check deps")
    ps.add_argument("--install-deps",action="store_true",help="Install deps")
    ps.add_argument("--pretty",action="store_true",help="Pretty JSON")
    a=ps.parse_args()
    if a.install_deps:
        pkgs=["Pillow","numpy","pytesseract","requests"]
        if not DEP["sklearn"]: pkgs.append("scikit-learn")
        if not DEP.get("insightface"): pkgs.extend(["insightface","opencv-python-headless"])
        subprocess.check_call([sys.executable,"-m","pip","install"]+pkgs,timeout=180); return
    if a.check:
        c={}
        c["pillow"]={"status":"ok","version":Image.__version__ if hasattr(Image,"__version__") else "?"} if DEP["pillow"] else {"status":"missing"}
        c["numpy"]={"status":"ok","version":np.__version__ if hasattr(np,"__version__") else "?"} if DEP["numpy"] else {"status":"missing"}
        c["pytesseract"]={"status":"ok"} if DEP["pytesseract"] else {"status":"missing"}
        if DEP["tesseract_bin"]:
            try: v=subprocess.check_output(["tesseract","--version"],stderr=subprocess.STDOUT,timeout=5).decode("utf-8",errors="replace").split("\n")[0].strip(); c["tesseract_binary"]={"status":"ok","version":v}
            except: c["tesseract_binary"]={"status":"ok","version":"?"}
        else: c["tesseract_binary"]={"status":"missing"}
        if DEP["pytesseract"] and DEP["tesseract_bin"]:
            try: c["tesseract_languages"]={"status":"ok","available":_pt.get_languages(config="")}
            except: pass
        c["ffmpeg"]={"status":"ok"} if DEP["ffmpeg"] else {"status":"missing"}
        c["requests"]={"status":"ok"} if DEP["requests"] else {"status":"missing"}
        c["sklearn"]={"status":"ok"} if DEP["sklearn"] else {"status":"missing"}
        c["siliconflow_api_key"]={"status":"ok"} if DEP["vision_api_key"] else {"status":"not set"}
        if DEP.get("insightface"):
            c["insightface"]={"status":"ok"}
            c["face_model"]={"status":"ok"} if DEP.get("face_model_loaded") else {"status":"missing_weights"}
        else: c["insightface"]={"status":"missing"}
        if DEP["vision_api_key"] and DEP["requests"]:
            try:
                r=_req.get("https://api.siliconflow.cn/v1/models",headers={"Authorization":f"Bearer {SILICONFLOW_API_KEY}"},timeout=10)
                if r.status_code==200:
                    ms=r.json().get("data",[])
                    c["siliconflow_api"]={"status":"ok","models_count":len(ms),"vision_model_available":any("VL" in m.get("id","") for m in ms)}
            except: pass
        print(json.dumps(c,ensure_ascii=False,indent=2 if a.pretty else None))
        sys.exit(0)
    if not a.image: ps.print_help(); sys.exit(1)
    ext=os.path.splitext(a.image)[1].lower()
    if ext in _VIDEO_EXT or a.video:
        result=analyze_video(a.image,fps=a.video_fps,enable_face=a.face or True)
    else:
        result=analyze_image(a.image,ocr_lang=a.ocr_lang,enable_vision=a.vision,enable_face=a.face)
    print(json.dumps(result,ensure_ascii=False,indent=2 if a.pretty else None,cls=_NumpyEncoder))
    if "error" in result: sys.exit(1)
if __name__=="__main__": main()