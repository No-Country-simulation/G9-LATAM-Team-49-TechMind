# TechMind Backend

API REST del MVP de TechMind. Recibe el título y el texto de un contenido técnico y devuelve
una lista de palabras clave en formato JSON.

## Requisitos

- Java 21
- No es necesario instalar Maven: el repositorio incluye Maven Wrapper.

## Ejecutar el proyecto

En Windows:

```powershell
.\mvnw.cmd spring-boot:run
```

En Linux o macOS:

```bash
./mvnw spring-boot:run
```

La API queda disponible en `http://localhost:8080`.

## Extraer palabras clave

```http
POST /api/v1/keywords
Content-Type: application/json
```

Solicitud:

```json
{
  "title": "Introducción a Spring Boot",
  "text": "Spring Boot permite crear APIs REST con Java."
}
```

Respuesta `200 OK`:

```json
{
  "keywords": ["Spring", "Boot", "Introducción", "permite", "crear"]
}
```

Los campos `title` y `text` son obligatorios. El título admite hasta 200 caracteres y el texto
hasta 20000 caracteres.

Respuesta de validación `400 Bad Request`:

```json
{
  "code": "VALIDATION_ERROR",
  "message": "La solicitud contiene campos inválidos",
  "fieldErrors": [
    {
      "field": "text",
      "message": "El texto es obligatorio"
    }
  ]
}
```

## Estado de la integración con Ciencia de Datos

Por ahora se usa un extractor local sencillo para permitir el desarrollo y las pruebas del flujo
completo. La interfaz `KeywordExtractor` mantiene aislada esta implementación. Cuando el equipo
defina el contrato del modelo, el extractor local se sustituirá por un cliente HTTP sin cambiar el
contrato utilizado por Frontend.

## Pruebas

```powershell
.\mvnw.cmd test
```
