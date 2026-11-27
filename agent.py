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

claude = anthropic.Anthropic()  # usa ANTHROPIC_API_KEY
db = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

PLANNER_SYSTEM = """Eres director de producción audiovisual. Convierte la orden en un plan de planos.
Responde SOLO con JSON válido, sin texto extra ni backticks:
{"title": str, "style": str, "shots": [{"n": int, "description": str, "prompt": str, "duration": int}]}
- "prompt": en inglés, cinematográfico (lente, movimiento de cámara, luz, atmósfera), autosuficiente.
- "duration": segundos por plano (3-8).
- Mantén coherencia visual de personajes y estilo entre planos repitiendo sus descripciones clave."""


# ---------- Proveedores de vídeo ----------
class VideoProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, duration: int, out_path: pathlib.Path) -> pathlib.Path:
        """Genera un clip y lo guarda en out_path (mp4)."""


class MockProvider(VideoProvider):
    """Clip negro de prueba para validar todo el pipeline sin gastar créditos."""
    def generate(self, prompt, duration, out_path):
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=1280x720:d={duration}:r=24",
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

    def generate(self, prompt, duration, out_path):
        args = {"prompt": prompt, "duration": str(duration), **self.extra}
        result = self.fal.subscribe(self.model, arguments=args)
        url = result["video"]["url"]
        urllib.request.urlretrieve(url, out_path)
        return out_path


def get_provider() -> VideoProvider:
    return FalProvider() if os.getenv("VIDEO_PROVIDER") == "fal" else MockProvider()


# ---------- Pasos del pipeline ----------
def plan_order(brief: str) -> dict:
    msg = claude.messages.create(
        model=MODEL, max_tokens=4000, system=PLANNER_SYSTEM,
        messages=[{"role": "user", "content": brief}])
    text = re.sub(r"^```(?:json)?|```$", "", msg.content[0].text.strip(), flags=re.M).strip()
    return json.loads(text)


def assemble(clips: list[pathlib.Path], out_path: pathlib.Path) -> pathlib.Path:
    list_file = out_path.with_suffix(".txt")
    list_file.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)],
        check=True, capture_output=True)
    return out_path


def run_order(order_id: str, brief: str):
    set_status = lambda **kw: db.table("orders").update(kw).eq("id", order_id).execute()
    try:
        set_status(status="planning")
        plan = plan_order(brief)
        set_status(plan=plan, status="generating")
        db.table("shots").insert([
            {"order_id": order_id, "n": s["n"], "prompt": s["prompt"],
             "duration": s["duration"], "status": "pending"} for s in plan["shots"]]).execute()

        provider, folder, clips = get_provider(), OUT / order_id, []
        folder.mkdir(parents=True, exist_ok=True)
        for s in plan["shots"]:
            clip = provider.generate(s["prompt"], s["duration"], folder / f"shot_{s['n']:02d}.mp4")
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

@app.post("/orders")
def create_order(order: Order, bg: BackgroundTasks, x_token: str = Header(default="")):
    if os.getenv("APP_TOKEN") and x_token != os.getenv("APP_TOKEN"):
        raise HTTPException(401, "Contraseña incorrecta")
    row = db.table("orders").insert({"brief": order.brief, "status": "queued"}).execute().data[0]
    bg.add_task(run_order, row["id"], order.brief)
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


PAGE = """<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agente de vídeo</title>
<style>body{font-family:system-ui;margin:0;padding:16px;max-width:600px;margin:auto}
textarea,input,button{width:100%;box-sizing:border-box;font-size:16px;padding:12px;margin:6px 0;border-radius:8px;border:1px solid #888}
button{background:#111;color:#fff;border:0}video{width:100%;margin-top:12px}pre{white-space:pre-wrap}</style></head>
<body><h2>Agente de vídeo</h2>
<input id="tk" type="password" placeholder="Contraseña">
<textarea id="brief" rows="5" placeholder="Describe el vídeo que quieres..."></textarea>
<button onclick="send()">Crear vídeo</button>
<pre id="st"></pre><video id="v" controls style="display:none"></video>
<script>
async function send(){
  const st=document.getElementById('st'), v=document.getElementById('v'); v.style.display='none';
  st.textContent='Enviando...';
  const r=await fetch('/orders',{method:'POST',headers:{'Content-Type':'application/json','x-token':tk.value},
    body:JSON.stringify({brief:brief.value})});
  if(!r.ok){st.textContent='Error: '+(await r.text());return}
  const {id}=await r.json();
  const t=setInterval(async()=>{
    const o=await (await fetch('/orders/'+id)).json();
    const done=(o.shots||[]).filter(s=>s.status==='done').length;
    st.textContent='Estado: '+o.status+' ('+done+'/'+(o.shots||[]).length+' planos)'+(o.error?'\n'+o.error:'');
    if(o.status==='done'){clearInterval(t);v.src='/orders/'+id+'/video';v.style.display='block'}
    if(o.status==='error')clearInterval(t);
  },3000);
}
</script></body></html>"""

@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
