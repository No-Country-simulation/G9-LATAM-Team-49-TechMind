# Despliegue en Oracle Cloud Infrastructure (OCI)

TechMind usa dos servicios de OCI:

- **OCI Compute** — una instancia Linux que ejecuta los dos contenedores.
- **OCI Object Storage** — un bucket con los artefactos del modelo entrenado, que el contenedor descarga al arrancar.

## Arquitectura desplegada

```text
                    Internet
                       │
                    :80 │ (único puerto abierto)
                       ▼
        ┌──────────────────────────────┐
        │  OCI Compute (VM.Standard)   │
        │                              │
        │  ┌────────────────────────┐  │
        │  │ frontend (nginx)       │  │
        │  │  · sitio Astro estático│  │
        │  │  · proxy /api → api    │  │
        │  └───────────┬────────────┘  │
        │              │ red interna   │
        │  ┌───────────▼────────────┐  │
        │  │ api (FastAPI :8000)    │  │
        │  │  · no expuesta fuera   │  │
        │  └───────────┬────────────┘  │
        └──────────────┼───────────────┘
                       │ al arrancar
                       ▼
            ┌──────────────────────┐
            │ OCI Object Storage   │
            │  bucket techmind-... │
            │  models/v2.0.0/*.job │
            └──────────────────────┘
```

Ventajas de esta topología frente a exponer el puerto 8000:

- Un único puerto que abrir en la Security List **y** en el firewall del sistema operativo.
- El navegador y la API comparten origen, así que CORS deja de ser una fuente de fallos.
- La API no queda expuesta a internet.

---

## 1. La instancia de Compute

### Shape

| Shape | RAM | ¿Sirve? |
|---|---|---|
| VM.Standard.E2.1.Micro | 1 GB | ❌ **No.** torch + sentence-transformers + BERTopic no caben; el contenedor muere por OOM |
| VM.Standard.A1.Flex (2 OCPU / 12 GB) | 12 GB | ✅ Recomendado. Está dentro del *Always Free* |
| VM.Standard.E4.Flex (1 OCPU / 8 GB) | 8 GB | ✅ De pago |

Imagen: Oracle Linux 8/9 (usuario `opc`) o Ubuntu 22.04 (usuario `ubuntu`). Disco de arranque: **al menos 50 GB** — las imágenes de Docker con torch ocupan varios GB.

### Instalar Docker

Oracle Linux:

```bash
sudo dnf install -y dnf-utils
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker $USER      # cerrar sesión y volver a entrar
```

Ubuntu:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```

### Abrir el puerto 80

Son **dos** cortafuegos independientes. Olvidar el segundo es el fallo de despliegue más común en OCI.

**a) Security List / NSG** (consola de OCI → Networking → VCN → Subnet → Security List → Add Ingress Rule):

| Campo | Valor |
|---|---|
| Source Type | CIDR |
| Source CIDR | `0.0.0.0/0` |
| IP Protocol | TCP |
| Destination Port Range | `80` |

**b) Firewall del sistema operativo**, dentro de la VM:

```bash
# Oracle Linux
sudo firewall-cmd --permanent --add-port=80/tcp
sudo firewall-cmd --reload
sudo iptables -I INPUT 6 -p tcp --dport 80 -j ACCEPT
sudo service iptables save 2>/dev/null || sudo netfilter-persistent save

# Ubuntu
sudo iptables -I INPUT 6 -p tcp --dport 80 -j ACCEPT
sudo netfilter-persistent save
```

Comprobación **desde fuera** de la VM (no desde dentro: desde dentro siempre parece funcionar):

```bash
curl -I http://<IP_PUBLICA>/
```

---

## 2. Object Storage: los artefactos del modelo

Los artefactos entrenados (varios MB de `.joblib`) no se versionan en Git. El bucket es su origen de verdad.

### Crear el bucket

Consola de OCI → Storage → Buckets → **Create Bucket**

- Nombre: `techmind-models`
- Storage Tier: Standard
- Visibilidad: **privada** (no marcar acceso público)

Anota el **namespace** del tenancy: aparece en los detalles del bucket como *Namespace*.

### Subir los artefactos

Desde la máquina donde entrenaste, con la [CLI de OCI](https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/cliinstall.htm) configurada (`oci setup config`):

```bash
oci os object bulk-upload \
  --bucket-name techmind-models \
  --src-dir ./models \
  --object-prefix models/v2.0.0/ \
  --content-type auto \
  --overwrite
```

Verificar:

```bash
oci os object list --bucket-name techmind-models --prefix models/v2.0.0/
```

Deben aparecer los seis: `metadata.json`, `modelo_clasificacion.joblib`, `label_encoder.joblib`, `config.json`, `centroides_clase.joblib`, `tecnologias.json`.

### Dar acceso al contenedor — opción A: PAR (recomendada para el hackathon)

Una **Pre-Authenticated Request** es una URL firmada con caducidad. No necesita SDK, credenciales ni políticas IAM.

Consola → Bucket → **Pre-Authenticated Requests** → *Create Pre-Authenticated Request*:

| Campo | Valor |
|---|---|
| Name | `techmind-models-read` |
| PAR Type | **Objects with prefix** |
| Prefix | `models/v2.0.0/` |
| Access Type | *Permit object reads on those with the specified prefix* |
| Expiration | una fecha posterior a la presentación |

O por CLI:

```bash
oci os preauth-request create \
  --bucket-name techmind-models \
  --name techmind-models-read \
  --access-type AnyObjectRead \
  --object-name models/v2.0.0/ \
  --time-expires 2026-12-31T23:59:00Z
```

⚠️ **La URL completa solo se muestra una vez.** Cópiala en ese momento. Tiene esta forma:

```
https://objectstorage.<region>.oraclecloud.com/p/<token>/n/<namespace>/b/techmind-models/o/models/v2.0.0/
```

En la VM, en el `.env`:

```bash
OCI_PAR_URL=https://objectstorage.sa-saopaulo-1.oraclecloud.com/p/XXXX/n/mi-namespace/b/techmind-models/o/models/v2.0.0/
```

La URL es un secreto: quien la tenga puede leer esos objetos. Nunca la subas al repositorio — el `.env` está en `.gitignore`.

### Dar acceso al contenedor — opción B: Instance Principals (producción)

La instancia se autentica con su propia identidad. Sin secretos en disco, y sin caducidad que gestionar.

1. **Dynamic Group** — Identity & Security → Domains → Dynamic Groups → Create:

   ```
   Name: techmind-instances
   Rule: instance.id = 'ocid1.instance.oc1..EL_OCID_DE_TU_INSTANCIA'
   ```

   O por compartimento entero:

   ```
   ALL {instance.compartment.id = 'ocid1.compartment.oc1..XXXX'}
   ```

2. **Policy** — Identity & Security → Policies → Create:

   ```
   Allow dynamic-group techmind-instances to read objects in compartment <nombre-compartimento> where target.bucket.name = 'techmind-models'
   ```

3. Descomentar `oci>=2.126.0` en `backend/requirements.txt` y reconstruir la imagen.

4. En el `.env` de la VM:

   ```bash
   OCI_NAMESPACE=mi-namespace
   OCI_BUCKET=techmind-models
   OCI_PREFIX=models/v2.0.0/
   ```

La lógica de descarga vive en `backend/services/oci_storage.py`. Si los artefactos ya están en `models/`, no descarga nada: el despliegue clásico (entrenar en la VM y montar la carpeta) sigue funcionando.

---

## 3. Desplegar

En la VM:

```bash
git clone https://github.com/No-Country-simulation/G9-LATAM-Team-49-TechMind.git
cd G9-LATAM-Team-49-TechMind

cp .env.example .env
nano .env      # rellenar OCI_PAR_URL (o OCI_NAMESPACE/OCI_BUCKET) y CORS_ORIGINS

docker compose up --build -d
docker compose logs -f api          # esperar a "Modelos precargados correctamente"
```

El primer arranque tarda varios minutos: construye la imagen con torch, descarga el modelo de embeddings de HuggingFace y carga spaCy. El volumen `hf_cache` evita repetir la descarga en arranques posteriores.

### Comprobación

Desde la VM:

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

Desde fuera:

```bash
IP=<IP_PUBLICA>
curl -s http://$IP/health
curl -s -X POST http://$IP/api/v1/contenido \
  -H "Content-Type: application/json" \
  -d '{"titulo":"Introduccion a Spring Boot","texto":"En este contenido se presentan los conceptos basicos para la creacion de APIs REST utilizando Java y Spring Boot."}'
```

---

## 4. Operación

| Acción | Comando |
|---|---|
| Ver estado | `docker compose ps` |
| Logs de la API | `docker compose logs -f api` |
| Consumo de recursos | `docker stats` |
| Reiniciar solo la API | `docker compose restart api` |
| Desplegar cambios | `git pull && docker compose up --build -d` |
| Liberar disco | `docker system prune -af --volumes` (⚠️ borra el caché de HuggingFace) |

### Si el contenedor `api` se reinicia solo

Casi siempre es falta de memoria:

```bash
docker compose logs api | tail -50
dmesg | grep -i "killed process"       # confirma un OOM kill
free -m
```

Con menos de 4 GB libres, este stack no arranca de forma estable. Cambia el shape de la instancia.

### Añadir HTTPS (opcional)

Si tienes un dominio apuntando a la IP, `certbot` con el plugin de nginx emite el certificado. Requiere abrir también el puerto 443 en la Security List y en `iptables`.

---

## 5. Consideraciones

- **CORS**: nunca `*`. Con el proxy inverso el origen es el mismo, así que `CORS_ORIGINS` solo cubre el caso de llamadas directas a la API.
- **Secretos**: el `.env` no se versiona. La URL de PAR es un secreto.
- **Memoria**: mínimo 4 GB libres; recomendado 8 GB.
- **Arranque**: `PRECARGAR_MODELOS=1` carga los modelos al iniciar. El `healthcheck` del `docker-compose.yml` da 300 s de margen (`start_period`) antes de considerar el contenedor enfermo.
