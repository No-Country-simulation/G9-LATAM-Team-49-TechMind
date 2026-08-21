import type { ContenidoRequest, ContenidoResponse } from '../types/techmind';

// Vacio => rutas relativas (mismo origen). Es lo correcto cuando nginx hace
// de proxy inverso hacia la API: el navegador llama a /api/v1/... en el
// puerto 80 y no hay que abrir el 8000 ni configurar CORS.
// En desarrollo local sin Docker: PUBLIC_API_URL=http://localhost:8000
const API_URL = import.meta.env.PUBLIC_API_URL ?? '';

/** Mensaje legible a partir del cuerpo de error de FastAPI. */
function mensajeDeError(datos: any, estado: number): string {
  const detalle = datos?.detail;

  if (Array.isArray(detalle)) {
    return detalle
      .map((d: any) => d?.mensaje ?? d?.msg ?? JSON.stringify(d))
      .join(' · ');
  }
  if (typeof detalle === 'string') return detalle;

  if (estado === 503) {
    return 'El modelo todavia no esta disponible en el servidor. Reintenta en unos segundos.';
  }
  return `El servidor respondio con el codigo ${estado}.`;
}

export async function processContent(payload: ContenidoRequest): Promise<ContenidoResponse> {
  const controlador = new AbortController();
  // La primera inferencia carga spaCy, KeyBERT y BERTopic: puede tardar.
  const temporizador = setTimeout(() => controlador.abort(), 120_000);

  try {
    const response = await fetch(`${API_URL}/api/v1/contenido`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controlador.signal,
    });

    if (!response.ok) {
      let datos: any = null;
      try {
        datos = await response.json();
      } catch {
        /* el cuerpo no era JSON */
      }
      throw new Error(mensajeDeError(datos, response.status));
    }

    return await response.json();
  } catch (error: any) {
    if (error?.name === 'AbortError') {
      throw new Error('La peticion supero el tiempo maximo de espera (120 s).');
    }
    console.error('Error de API:', error);
    throw error;
  } finally {
    clearTimeout(temporizador);
  }
}
