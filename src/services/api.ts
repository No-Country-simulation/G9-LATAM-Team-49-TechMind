import type { ContenidoRequest, ContenidoResponse } from '../types/techmind';

const API_URL = import.meta.env.PUBLIC_API_URL || 'http://localhost:8000';

export async function processContent(payload: ContenidoRequest): Promise<ContenidoResponse> {
  try {
    const response = await fetch(`${API_URL}/api/v1/contenido`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail ? JSON.stringify(errorData.detail) : 'Failed to process content');
    }

    return await response.json();
  } catch (error) {
    console.error("API Error:", error);
    throw error;
  }
}
