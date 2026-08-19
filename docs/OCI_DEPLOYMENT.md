# Despliegue en Oracle Cloud Infrastructure (OCI)

Este proyecto está diseñado para desplegarse fácilmente en OCI utilizando instancias de Compute y (opcionalmente) contenedores.

## Arquitectura

1. **Frontend (Astro)**: Se construirá estáticamente (`Astro Static`).
2. **Backend (FastAPI)**: Se ejecutará utilizando Docker o directamente en una instancia de Linux.

## Pasos de Despliegue

### 1. Preparar el Backend (Instancia OCI Compute)
1. Crear una Instancia en OCI (Ubuntu 22.04 recomendado).
2. Clonar el repositorio en la instancia.
3. Copiar el `.env.example` a `.env` y configurar `CORS_ORIGINS` para que apunte al dominio donde vivirá el frontend (ej. `https://frontend.midominio.com`).
4. Construir y ejecutar el contenedor Docker:
   ```bash
   docker build -t techmind-backend .
   docker run -d -p 8000:8000 --env-file .env techmind-backend
   ```
5. Asegurar que las reglas de red (Security Lists / NSG) de OCI permitan tráfico TCP en el puerto 8000.
6. (Opcional) Configurar Nginx como proxy inverso para exponer FastAPI por HTTPS en el puerto 443.

### 2. Preparar el Frontend (Astro)
1. En tu máquina local o pipeline CI/CD, establecer la variable de entorno de producción apuntando a la IP pública de la instancia de OCI o a su dominio:
   ```bash
   PUBLIC_API_URL=http://<IP_PUBLICA_OCI>:8000
   ```
2. Construir el frontend:
   ```bash
   cd frontend
   npm run build
   ```
3. El resultado estará en la carpeta `frontend/dist/`.
4. Alojar el contenido estático generado en cualquier servicio (ej. OCI Object Storage configurado como sitio estático web, Netlify, Vercel, o dentro del mismo Nginx en la instancia OCI).

### 3. Consideraciones de Producción
*   **Modelos ML**: Asegurar que la instancia de Compute tenga suficiente memoria (recomendado al menos 8GB RAM) para cargar en memoria BERT y KeyBERT sin penalizar el rendimiento.
*   **CORS**: Jamás usar `*` en producción. Configurar estrictamente la URL final estática.
