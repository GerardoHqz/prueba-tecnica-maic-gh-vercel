import logging
from contextlib import asynccontextmanager
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.config import settings
from app.database import close_db, connect_db
from app.services.ai_service import analizar_mensaje
from app.services.message_service import (
    actualizar_analisis_mensaje,
    guardar_mensaje,
    listar_mensajes,
    obtener_sentimientos,
    obtener_temas,
)

logger = logging.getLogger(__name__)

# administra la conexión a MongoDB
@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    yield
    await close_db()


app = FastAPI(
    title="Café El Salvador API",
    description="API para mensajes de WhatsApp y análisis de sentimiento",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


def _parse_twilio_form(body: bytes) -> dict[str, str]:
    """Convierte body application/x-www-form-urlencoded en dict str -> str (primer valor por clave)."""
    decoded = body.decode("utf-8")
    parsed = parse_qs(decoded, keep_blank_values=True)
    return {k: (v[0] if v else "") for k, v in parsed.items()}


"""
Webhook que Twilio llama cuando llega un mensaje al número de WhatsApp.
Guarda en MongoDB: texto_mensaje, numero_remitente, timestamp.
"""
@app.post("/webhook/whatsapp")
async def webhook_whatsapp(request: Request):
    body_bytes = await request.body()
    form = _parse_twilio_form(body_bytes)

    body_text = (form.get("Body") or "").strip()
    from_number = form.get("From") or ""

    if not body_text:
        raise HTTPException(status_code=400, detail="El cuerpo del mensaje no puede estar vacío")
    if not from_number:
        raise HTTPException(status_code=400, detail="Falta el número remitente (From)")

    mensaje, mensaje_id = await guardar_mensaje(texto_mensaje=body_text, numero_remitente=from_number)
    if settings.GEMINI_API_KEY:
        try:
            resultado = await analizar_mensaje(body_text)
            await actualizar_analisis_mensaje(
                mensaje_id,
                sentimiento=resultado.analisis.sentimiento,
                tema=resultado.analisis.tema,
                resumen=resultado.analisis.resumen,
                ai_model_id=resultado.model_id,
                ai_latency_ms=resultado.latency_ms,
                prompt_version=resultado.prompt_version,
            )
        except Exception as e:
            logger.exception("Error al analizar mensaje con IA: %s", e)
    else:
        logger.warning("GEMINI_API_KEY no configurada: mensaje guardado sin análisis.")
    # Twilio acepta 200 OK; respuesta vacía o TwiML. Devolvemos 200 sin cuerpo.
    return Response(content="", status_code=200)

@app.get("/api/mensajes", response_model=list)
async def ver_mensajes(limite: int = 100):
    """Lista los mensajes recibidos por WhatsApp (más recientes primero)."""
    return await listar_mensajes(limite=limite)


@app.get("/api/sentimientos", response_model=list)
async def ver_sentimientos():
    """Conteo de mensajes por sentimiento (positivo, negativo, neutro, sin_clasificar)."""
    return await obtener_sentimientos()


@app.get("/api/temas", response_model=list)
async def ver_temas():
    """Conteo de mensajes por tema."""
    return await obtener_temas()