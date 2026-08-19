export interface ContenidoRequest {
  titulo: string;
  texto: string;
  n_keywords?: number;
  id_externo?: string;
}

export interface ContenidoResponse {
  categoria: string;
  probabilidad: number;
  informacion_adicional: any[];
  doc_id: string;
  tiempo_ms: number;
  titulo: string;
  idioma: Record<string, any>;
  tema: Record<string, any>;
  entidades_tecnicas: string[];
  distribucion_categorias: Record<string, number>;
  metricas_texto: Record<string, any>;
  explicacion: Record<string, any>;
  relacionados: Record<string, any>[];
  advertencias: Record<string, string>[];
}
