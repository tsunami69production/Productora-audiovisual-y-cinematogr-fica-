"""Agente de vídeo IA: orden -> plan de planos (Claude) -> clips (proveedor) -> montaje (ffmpeg)."""
import json, os, pathlib, re, subprocess, traceback, urllib.request
from abc import ABC, abstractmethod

import anthropic
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from supabase import create_client

MODEL = "claude-sonnet-5-5"
OUT = pathlib.Path(os.getenv("OUTPUT_DIR", "outputs"))
OUT.mkdir(exist_ok=True)

MAX_TOTAL = int(os.getenv("MAX_TOTAL_SECONDS", "60"))   # tope de duración por vídeo (controla el gasto)
CLIP_SECONDS = int(os.getenv("CLIP_SECONDS", "5"))      # duración aproximada de cada plano
ASPECTS = {"16:9": "1280x720", "9:16": "720x1280", "1:1": "720x720"}
RESOLUTIONS = ("480p", "720p", "1080p")

claude = anthropic.Anthropic()  # usa ANTHROPIC_API_KEY
db = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

PLANNER_SYSTEM = """Eres director de producción audiovisual. Convierte la orden en un plan de planos.
Responde SOLO con JSON válido, sin texto extra ni backticks:
{"title": str, "style": str, "shots": [{"n": int, "description": str, "prompt": str, "duration": int}]}
- Devuelve EXACTAMENTE el número de planos que indique el usuario.
- "prompt": en inglés, cinematográfico (lente, movimiento de cámara, luz, atmósfera), autosuficiente.
  Aplica el estilo pedido en cada prompt y compón pensando en el formato (vertical/horizontal/cuadrado).
- "duration": segundos por plano (los ajustará el sistema).
- Mantén coherencia visual de personajes y estilo entre planos repitiendo sus descripciones clave."""


# ---------- Proveedores de vídeo ----------
class VideoProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, duration: int, out_path: pathlib.Path, settings: dict) -> pathlib.Path:
        """Genera un clip y lo guarda en out_path (mp4)."""


class MockProvider(VideoProvider):
    """Clip negro de prueba para validar todo el pipeline sin gastar créditos."""
    def generate(self, prompt, duration, out_path, settings):
        size = ASPECTS.get(settings.get("aspect"), "1280x720")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s={size}:d={duration}:r=24",
             "-pix_fmt", "yuv420p", str(out_path)],
            check=True, capture_output=True)
        return out_path


class FalProvider(VideoProvider):
    """Genera clips con fal.ai. El modelo se elige con FAL_MODEL (ID exacto de su página en fal.ai),
    p. ej. 'bytedance/seedance-2.0/text-to-video'. Usa FAL_KEY."""
    def __init__(self):
        import fal_client
        self.fal = fal_client
        self.model = os.environ["FAL_MODEL"]
        self.extra = json.loads(os.getenv("FAL_EXTRA_ARGS", "{}"))  # p. ej. {"resolution":"720p"}

    def generate(self, prompt, duration, out_path, settings):
        args = {"prompt": prompt, "duration": str(duration)}
        if os.getenv("FAL_SEND_SETTINGS", "1") == "1":  # pon 0 si tu modelo rechaza estos parámetros
            args["aspect_ratio"] = settings.get("aspect", "16:9")
            args["resolution"] = settings.get("resolution", "720p")
        args.update(self.extra)
        result = self.fal.subscribe(self.model, arguments=args)
        url = result["video"]["url"]
        urllib.request.urlretrieve(url, out_path)
        return out_path


def get_provider() -> VideoProvider:
    return FalProvider() if os.getenv("VIDEO_PROVIDER") == "fal" else MockProvider()


# ---------- Pasos del pipeline ----------
def shot_count(total: int) -> int:
    return max(1, round(total / CLIP_SECONDS))


def plan_order(brief: str, settings: dict) -> dict:
    k = shot_count(settings["duration"])
    user = (f"Orden: {brief}\n\nAjustes: duración total {settings['duration']} s; "
            f"número de planos: {k}; formato: {settings['aspect']}; estilo: {settings['style']}.")
    msg = claude.messages.create(
        model=MODEL, max_tokens=4000, system=PLANNER_SYSTEM,
        messages=[{"role": "user", "content": user}])
    text = re.sub(r"^```(?:json)?|```$", "", msg.content[0].text.strip(), flags=re.M).strip()
    plan = json.loads(text)
    shots = plan["shots"][:k]                      # recorta si Claude devolvió de más
    base, rem = divmod(settings["duration"], len(shots))
    for i, sh in enumerate(shots):                 # reparte los segundos totales entre los planos
        sh["n"] = i + 1
        sh["duration"] = base + (1 if i < rem else 0)
    plan["shots"], plan["settings"] = shots, settings
    return plan


def assemble(clips: list[pathlib.Path], out_path: pathlib.Path) -> pathlib.Path:
    list_file = out_path.with_suffix(".txt")
    list_file.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)],
        check=True, capture_output=True)
    return out_path


def run_order(order_id: str, brief: str, settings: dict):
    set_status = lambda **kw: db.table("orders").update(kw).eq("id", order_id).execute()
    try:
        set_status(status="planning")
        plan = plan_order(brief, settings)
        set_status(plan=plan, status="generating")
        db.table("shots").insert([
            {"order_id": order_id, "n": s["n"], "prompt": s["prompt"],
             "duration": s["duration"], "status": "pending"} for s in plan["shots"]]).execute()

        provider, folder, clips = get_provider(), OUT / order_id, []
        folder.mkdir(parents=True, exist_ok=True)
        for s in plan["shots"]:
            clip = provider.generate(s["prompt"], s["duration"], folder / f"shot_{s['n']:02d}.mp4", settings)
            clips.append(clip)
            db.table("shots").update({"status": "done", "path": str(clip)}) \
              .eq("order_id", order_id).eq("n", s["n"]).execute()

        set_status(status="assembling")
        final = assemble(clips, folder / "final.mp4")
        set_status(status="done", result_path=str(final))
    except Exception:
        set_status(status="error", error=traceback.format_exc()[-2000:])


# ---------- API ----------
app = FastAPI()

class Order(BaseModel):
    brief: str
    duration: int = 15
    aspect: str = "16:9"
    resolution: str = "720p"
    style: str = "cinematográfico"

@app.post("/orders")
def create_order(order: Order, bg: BackgroundTasks, x_token: str = Header(default="")):
    if os.getenv("APP_TOKEN") and x_token != os.getenv("APP_TOKEN"):
        raise HTTPException(401, "Contraseña incorrecta")
    try:
        row = db.table("orders").insert({"brief": order.brief, "status": "queued"}).execute().data[0]
    except Exception as e:
        raise HTTPException(500, f"No se pudo guardar en Supabase: {str(e)[:300]}")
    settings = {
        "duration": max(3, min(order.duration, MAX_TOTAL)),
        "aspect": order.aspect if order.aspect in ASPECTS else "16:9",
        "resolution": order.resolution if order.resolution in RESOLUTIONS else "720p",
        "style": order.style.strip()[:60] or "cinematográfico",
    }
    bg.add_task(run_order, row["id"], order.brief, settings)
    return {"id": row["id"], "status": "queued"}

@app.get("/orders/{order_id}")
def get_order(order_id: str):
    order = db.table("orders").select("*").eq("id", order_id).single().execute().data
    order["shots"] = db.table("shots").select("*").eq("order_id", order_id).order("n").execute().data
    return order


@app.get("/orders/{order_id}/video")
def get_video(order_id: str):
    path = OUT / order_id / "final.mp4"
    if not path.exists():
        raise HTTPException(404, "Vídeo no disponible")
    return FileResponse(path, media_type="video/mp4")


BRAND = os.getenv("BRAND_NAME", "Tsunami_69 Production")
SITE_URL = os.getenv("SITE_URL", "")

PAGE = """<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#050b18">
<title>__BRAND__</title>
<link rel="icon" href="/logo.png">
<style>
:root{--bg:#050b18;--card:#0c1730;--line:#1b2c52;--txt:#e8f1ff;--mut:#8aa0c8;--a:#19e3d0;--b:#2f7bff}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--txt);
background:radial-gradient(1200px 600px at 50% -10%,#10285a 0%,var(--bg) 60%);overflow-x:hidden}
.wrap{max-width:560px;margin:auto;padding:22px 18px 90px;position:relative;z-index:2}
header{display:none;align-items:center;gap:12px}
.logo{width:54px;height:54px;border-radius:14px;object-fit:contain}
#lg{display:block}
.brand{font-weight:800;letter-spacing:.18em;text-transform:uppercase;font-size:15px;line-height:1.25}
.brand small{display:block;letter-spacing:.3em;color:var(--a);font-weight:600;font-size:10px;margin-top:2px}
.hero{display:block;width:100%;max-width:440px;margin:6px auto 0;border-radius:22px;border:1px solid var(--line);box-shadow:0 0 50px #19e3d033,0 12px 40px #000a}
h1{font-size:28px;text-align:center;line-height:1.1;margin:22px 0 8px}
h1 span{background:linear-gradient(90deg,var(--a),var(--b));-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{color:var(--mut);margin:0 0 22px;font-size:15px;text-align:center}
.card{background:linear-gradient(180deg,#0e1b38,var(--card));border:1px solid var(--line);border-radius:18px;padding:16px;box-shadow:0 10px 40px #0008}
label{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--mut)}
input,textarea{width:100%;font:inherit;font-size:16px;color:var(--txt);background:#071027;border:1px solid var(--line);
border-radius:12px;padding:12px;margin:6px 0 14px;outline:none}
input:focus,textarea:focus{border-color:var(--a)}
button{width:100%;font:inherit;font-size:17px;font-weight:700;color:#04101f;border:0;border-radius:12px;padding:14px;
background:linear-gradient(90deg,var(--a),var(--b));cursor:pointer}
button:disabled{opacity:.5}
.track{height:8px;background:#0a1530;border-radius:99px;margin-top:18px;overflow:hidden;border:1px solid var(--line)}
#bar{height:100%;width:0;background:linear-gradient(90deg,var(--a),var(--b));transition:width .6s}
#st{margin:10px 2px 0;color:var(--mut);font-size:14px;white-space:pre-wrap;min-height:20px}
video{width:100%;margin-top:14px;border-radius:14px;border:1px solid var(--line);background:#000;display:none}
#dl{display:none;margin-top:10px;text-align:center;color:var(--a);font-weight:600;text-decoration:none}
select{width:100%;font:inherit;font-size:16px;color:var(--txt);background:#071027;border:1px solid var(--line);border-radius:12px;padding:12px;margin:6px 0 14px}
input[type=range]{padding:0;border:0;background:none;accent-color:var(--a);margin:10px 0 14px}
.opts{border:1px solid var(--line);border-radius:14px;padding:12px 14px 2px;margin:0 0 16px;background:#08122a}
.opts summary{cursor:pointer;font-weight:700;margin-bottom:12px;color:var(--a)}
.opts label b{color:var(--txt);letter-spacing:0;text-transform:none;font-size:14px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
footer{margin-top:34px;text-align:center;color:var(--mut);font-size:13px}
footer a{color:var(--a);text-decoration:none;font-weight:600}
.waves{position:fixed;left:0;right:0;bottom:0;height:90px;z-index:1;pointer-events:none;opacity:.55}
.waves svg{position:absolute;bottom:0;width:200%;height:100%;animation:mv 14s linear infinite}
.waves svg:nth-child(2){animation-duration:22s;opacity:.6;animation-direction:reverse}
@keyframes mv{from{transform:translateX(0)}to{transform:translateX(-50%)}}
</style></head>
<body>
<div class="wrap">
<img class="hero" src="/logo.png" alt="__BRAND__" onerror="this.style.display='none';document.querySelector('header').style.display='flex'">
<header>
<svg id="lg" class="logo" viewBox="0 0 64 64"><defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#19e3d0"/><stop offset="1" stop-color="#2f7bff"/></linearGradient></defs>
<rect width="64" height="64" rx="16" fill="#0c1730" stroke="#1b2c52"/>
<path d="M8 40c6-14 14-20 24-18 8 2 8 10 2 12-5 2-9-2-6-6 4 6 13 6 18-2 2 8-4 20-18 20H8z" fill="url(#g)"/></svg>
<div class="brand">__BRAND__<small>AUDIOVISUAL &amp; CINE IA</small></div>
</header>
<h1>Convierte tus ideas en <span>cine</span></h1>
<p class="sub">Describe la escena y el agente escribe el guion, planifica los planos y monta el vídeo.</p>
<div class="card">
<label for="tk">Contraseña</label>
<input id="tk" type="password" autocomplete="current-password" placeholder="Tu contraseña de acceso">
<label for="brief">Tu idea</label>
<textarea id="brief" rows="5" placeholder="Ej.: Spot de 30 s para una marca de café, estilo noir, lluvia nocturna..."></textarea>
<details class="opts" open><summary>Ajustes del vídeo</summary>
<label for="dur">Duración: <b id="durv">15 s</b> <span id="shn"></span></label>
<input id="dur" type="range" min="5" max="__MAXDUR__" step="1" value="15" oninput="upd()">
<div class="row">
<div><label for="asp">Formato</label>
<select id="asp"><option value="16:9">16:9 Horiz.</option><option value="9:16">9:16 Vertical</option><option value="1:1">1:1 Cuadrado</option></select></div>
<div><label for="res">Resolución</label>
<select id="res"><option>480p</option><option selected>720p</option><option>1080p</option></select></div>
</div>
<label for="sty">Estilo</label>
<select id="sty"><option>cinematográfico</option><option>cine negro (noir)</option><option>publicitario / spot comercial</option><option>documental</option><option>animación 3D</option><option>anime</option><option>ciencia ficción</option><option>fantasía épica</option><option>videoclip musical</option></select>
</details>
<button id="go" onclick="send()">Crear vídeo</button>
<div class="track"><div id="bar"></div></div>
<div id="st"></div>
<video id="v" controls playsinline></video>
<a id="dl" download="video.mp4">Descargar vídeo</a>
</div>
<footer>© <span id="yr"></span> __BRAND__ __SITE__</footer>
</div>
<div class="waves">
<svg viewBox="0 0 1200 90" preserveAspectRatio="none"><path d="M0 45C100 10 200 80 300 45S500 10 600 45 800 80 900 45 1100 10 1200 45V90H0z" fill="#2f7bff" opacity=".35"/></svg>
<svg viewBox="0 0 1200 90" preserveAspectRatio="none"><path d="M0 55C100 25 200 85 300 55S500 25 600 55 800 85 900 55 1100 25 1200 55V90H0z" fill="#19e3d0" opacity=".3"/></svg>
</div>
<script>
const $=id=>document.getElementById(id);
$('yr').textContent=new Date().getFullYear();
function upd(){const d=+$('dur').value;$('durv').textContent=d+' s';$('shn').textContent='· ≈'+Math.max(1,Math.round(d/__CLIP__))+' planos'}
upd();
try{$('tk').value=localStorage.getItem('tk')||''}catch(e){}
async function send(){
  const st=$('st'),bar=$('bar'),v=$('v'),dl=$('dl'),btn=$('go');
  v.style.display='none';dl.style.display='none';
  if(!$('brief').value.trim()){st.textContent='Escribe primero la idea del vídeo.';return}
  try{localStorage.setItem('tk',$('tk').value)}catch(e){}
  btn.disabled=true;st.textContent='Enviando orden...';bar.style.width='4%';
  try{
    const r=await fetch('/orders',{method:'POST',headers:{'Content-Type':'application/json','x-token':$('tk').value},
      body:JSON.stringify({brief:$('brief').value,duration:+$('dur').value,aspect:$('asp').value,resolution:$('res').value,style:$('sty').value})});
    if(!r.ok){st.textContent='Error: '+(await r.text());btn.disabled=false;bar.style.width='0';return}
    const {id}=await r.json();
    const t=setInterval(async()=>{
      try{
        const rr=await fetch('/orders/'+id); if(!rr.ok)return;
        const o=await rr.json();
        const n=(o.shots||[]).length, done=(o.shots||[]).filter(s=>s.status==='done').length;
        const names={queued:'En cola',planning:'Escribiendo guion y plano a plano',generating:'Generando planos ('+done+'/'+n+')',assembling:'Montando el vídeo',done:'Listo',error:'Error'};
        let p={queued:5,planning:15,assembling:92,done:100,error:0}[o.status];
        if(o.status==='generating')p=15+(n?75*done/n:0);
        bar.style.width=p+'%';
        st.textContent=(names[o.status]||o.status)+(o.error?' - '+o.error:'');
        if(o.status==='done'){clearInterval(t);v.src='/orders/'+id+'/video';v.style.display='block';
          dl.href=v.src;dl.style.display='block';btn.disabled=false}
        if(o.status==='error'){clearInterval(t);btn.disabled=false}
      }catch(e){}
    },3000);
  }catch(e){st.textContent='Error de conexión: '+e;btn.disabled=false;bar.style.width='0'}
}
</script></body></html>"""


@app.get("/logo.png")
def logo():
    p = pathlib.Path("logo.png")
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.get("/", response_class=HTMLResponse)
def home():
    site = f'· <a href="{SITE_URL}">{SITE_URL.replace("https://", "").replace("http://", "").rstrip("/")}</a>' if SITE_URL else ""
    return (PAGE.replace("__BRAND__", BRAND).replace("__SITE__", site)
            .replace("__MAXDUR__", str(MAX_TOTAL)).replace("__CLIP__", str(CLIP_SECONDS)))
