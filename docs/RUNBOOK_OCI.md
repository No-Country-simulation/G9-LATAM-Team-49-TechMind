# TechMind — runbook de despliegue en OCI

**Fecha:** 21 de agosto de 2026
**Repositorio validado:** `main` en `af967c5`
**Acompaña a:** `techmind-oci.zip`

---

## Estado verificado hoy

He vuelto a clonar el repositorio y a ejecutarlo. Esto es lo que hay:

| Comprobación | Resultado |
|---|---|
| `main` | `af967c5` — "fix: restaurar imports tras el refactor…" ✅ **subido** |
| Etiqueta de seguridad | `estado-19-agosto` presente en el remoto ✅ |
| `pytest tests/ -v` | **13 passed** ✅ |
| `nginx.conf` | Sintaxis validada con `nginx -t` ✅ |
| `docker-compose.yml` | Puerto 80 público, API en loopback ✅ |
| `.dockerignore` | Presente ✅ |
| **Ronda 2** | ❌ **no aplicada** |

La ronda 1 está en el repositorio y funciona. Pero **dos de los tres arreglos de la ronda 2 son precisamente bloqueantes de OCI**, así que el bloque 0 de este runbook es aplicarlos. Sin ellos el despliegue arranca y engaña: `/health` responde `200` y el análisis funciona, pero faltan dos de las cinco capacidades del brief.

Lo que sigue abierto, y cómo se manifiesta **en la VM**:

| Fichero en `main` hoy | Qué pasa en OCI |
|---|---|
| `backend/services/oci_storage.py` descarga 6 ficheros sueltos a `models/` | `models/modelo_bertopic/` y `chroma_db/` nunca llegan → `tema` sale `{"id": -1}` y `relacionados` sale `[]` en **todas** las respuestas |
| `backend/Dockerfile` → `python:3.10-slim` + rangos `>=` | Entrenaste con Python 3.12.10 y scikit-learn 1.9.0. Los `.joblib` son pickles: en la VM pueden dar `InconsistentVersionWarning` con predicciones distintas, o `AttributeError` y la API no arranca |
| Sin `requirements.lock.txt` | `pip` resuelve versiones distintas cada vez que reconstruyes: el despliegue no es reproducible |
| El modelo de embeddings se baja de HuggingFace al arrancar | Primer arranque lento y dependiente de la red; si HF aplica *rate limit* durante la demo, `/health` se queda en 503 |
| `.env.example` → `OCI_PREFIX=models/v2.0.0/` | Debe pasar a `paquete/v2.0.0/` con el empaquetado nuevo |

---

# Bloque 0 — Aplicar la ronda 2

**15 minutos. En tu PC, antes de tocar la VM.**

```powershell
cd "C:\Users\Luis Rodriguez\Desktop\OCI\G9-LATAM-Team-49-TechMind"
git pull origin main

# Copia el contenido de techmind-oci\ encima del repo
Copy-Item -Path "C:\Users\Luis Rodriguez\Downloads\techmind-oci\*" `
          -Destination . -Recurse -Force

git diff --stat        # revisa antes de aceptar
```

Genera el fichero de versiones ancladas **desde el entorno donde entrenaste**, que es el único que sabemos que funciona:

```powershell
.\.venv\Scripts\Activate.ps1
pip freeze | Out-File -Encoding utf8 backend\requirements.lock.txt

# Quita rutas locales de Windows, que no valen dentro del contenedor
(Get-Content backend\requirements.lock.txt) `
  | Where-Object { $_ -notmatch '@ file:///' -and $_ -notmatch '^-e ' } `
  | Set-Content backend\requirements.lock.txt

Select-String -Path backend\requirements.lock.txt -Pattern "scikit-learn|numpy|pandas|spacy|bertopic"
```

Contrasta esas cinco líneas con tu log de entrenamiento: `numpy 2.5.2`, `pandas 3.0.5`, `sklearn 1.9.0`, `spacy 3.8.15`, `bertopic 0.17.4`.

Comprueba y sube:

```powershell
$env:PYTHONPATH="backend"; $env:PRECARGAR_MODELOS="0"
pytest tests\ -q                      # esperado: 13 passed

git add backend\ scripts\ docker-compose.yml .env.example
git commit -m "fix: paquete del modelo completo desde Object Storage y entorno reproducible"
git push origin main
```

---

# Bloque 1 — La instancia

**Antes de nada, mira el shape. Es el riesgo silencioso del proyecto.**

Conéctate:

```powershell
ssh -i C:\ruta\a\tu\clave.key opc@147.224.239.129
```

```bash
free -m ; nproc ; df -h / ; uname -m
```

## Qué shape necesitas

| Shape | RAM | ¿Sirve? |
|---|---|---|
| `VM.Standard.E2.1.Micro` | **1 GB** | ❌ **No.** torch + sentence-transformers + BERTopic no caben. Morirá por OOM y no hay optimización que lo arregle |
| `VM.Standard.A1.Flex` (2 OCPU / 12 GB) | 12 GB | ✅ Correcto, y entra en el *Always Free* |
| `VM.Standard.E4.Flex` (1 OCPU / 8 GB) | 8 GB | ✅ De pago |

El *Always Free* de Ampere A1 son **2 OCPU y 12 GB** (Oracle redujo a la mitad la asignación en 2026; antes eran 4 OCPU y 24 GB). Los volúmenes de arranque son 50 GB por instancia, con 200 GB combinados en total, y Object Storage da 20 GB y 50 000 peticiones de API al mes — de sobra para tu paquete de modelo, que son unos pocos MB.

> **Si al crear la A1 te sale "Out of host capacity"**, es la saturación habitual de Ampere en el *free tier*, no un error tuyo. Reintenta en otro *Availability Domain* o a otra hora. No te quedes bloqueado: si hay prisa, una `E4.Flex` de pago cuesta céntimos por el día de la hackathon.

## ⚠️ Si la instancia es Ampere (ARM)

`uname -m` devolviendo `aarch64` significa que estás en ARM. En ese caso, **edita `backend/Dockerfile` y quita el índice de PyTorch**:

```dockerfile
# En x86_64 (AMD/Intel):
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# En aarch64 (Ampere) — deja que pip lo resuelva desde PyPI:
RUN pip install --no-cache-dir torch
```

Ese índice existe para separar las variantes CUDA y CPU en x86_64. En ARM no hay variante CUDA, así que el `torch` de PyPI ya es solo-CPU y el índice especial solo te complica la resolución.

## Disco

El commit `23e62c5` de vuestro historial ("prevent out-of-disk error on OCI VM") dice que ya os topasteis con esto. La imagen con torch ocupa varios GB. Con menos de 30 GB libres vas justo; con menos de 15, no cabe.

```bash
df -h /
docker system df          # si ya hay imágenes viejas ocupando sitio
```

---

# Bloque 2 — Preparar la VM

## Docker

**Oracle Linux 8/9:**

```bash
sudo dnf install -y dnf-utils
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
exit          # cierra la sesión SSH y vuelve a entrar: el grupo no aplica hasta entonces
```

**Ubuntu 22.04:**

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
exit
```

## Clonar y comprobar

```bash
cd ~
git clone https://github.com/No-Country-simulation/G9-LATAM-Team-49-TechMind.git
cd G9-LATAM-Team-49-TechMind

cp .env.example .env
nano .env                 # lo rellenamos en el bloque 4

bash scripts/preflight_oci.sh
```

Ese script viene en el ZIP. Comprueba RAM, disco, arquitectura, Docker, cortafuegos, variables de entorno, presencia del paquete del modelo y salida a internet. Cada control corresponde a un fallo de despliegue real. **No sigas mientras dé FALLO.**

---

# Bloque 3 — Los dos cortafuegos

Este es el fallo de despliegue más común en OCI, y la razón es que hay **dos** cortafuegos independientes. La gente configura el primero, comprueba desde dentro de la VM —donde siempre parece funcionar— y descubre el segundo delante del jurado.

## a) Security List de OCI (consola web)

Networking → Virtual Cloud Networks → tu VCN → Subnets → tu subred → Security Lists → **Add Ingress Rules**:

| Campo | Valor |
|---|---|
| Stateless | No |
| Source Type | CIDR |
| Source CIDR | `0.0.0.0/0` |
| IP Protocol | TCP |
| Source Port Range | *(vacío)* |
| Destination Port Range | `80` |

## b) Cortafuegos del sistema operativo (dentro de la VM)

Oracle Linux trae `iptables` cerrado salvo el 22 por defecto:

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

Comprueba que la regla quedó:

```bash
sudo iptables -S INPUT | grep 80
```

**No hace falta abrir el 8000.** Con el proxy inverso de nginx, la API solo escucha en `127.0.0.1` dentro de la VM.

## Verificación real

Desde **tu PC**, no desde la VM:

```powershell
curl.exe -I -m 15 http://147.224.239.129/
```

Mientras esto no responda, no sigas: todo lo demás lo comprobarás a ciegas.

---

# Bloque 4 — Object Storage

Aquí es donde se cumple el requisito obligatorio de integración con OCI, y donde el contenedor consigue los artefactos que `.gitignore` mantiene fuera del repositorio.

## 4.1 Crear el bucket

Consola → Storage → Buckets → **Create Bucket**:

- Nombre: `techmind-models`
- Storage Tier: Standard
- Visibilidad: **privada** (no marques acceso público)

Anota el **Namespace** que aparece en los detalles del bucket.

## 4.2 Instalar la OCI CLI (en tu PC, no en la VM)

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
Invoke-WebRequest https://raw.githubusercontent.com/oracle/oci-cli/master/scripts/install/install.ps1 -OutFile install.ps1
.\install.ps1
# Reinicia PowerShell
oci setup config
```

`oci setup config` pide el OCID de tu usuario (consola → tu avatar → *User settings*), el del tenancy y la región, y genera un par de claves. Después hay que **subir la clave pública** en *User settings → API Keys → Add API Key*.

## 4.3 Empaquetar el modelo

**Este es el paso que arregla el bug de los seis ficheros.** El script recoge `models/` completo —incluido `modelo_bertopic/`— más `chroma_db/`, y escribe un `manifest.json` con la ruta, el tamaño y el hash de cada fichero:

```powershell
python scripts\preparar_paquete_modelo.py
```

Lee la tabla que imprime:

```
    chroma_db          N ficheros    XXX KB
    models            10 ficheros    XXX KB
```

Si `models` sale con **6** en vez de ~10, `modelo_bertopic/` no se generó: vuelve a entrenar antes de continuar. Ese es exactamente el fallo que deja el campo `tema` vacío en producción.

## 4.4 Subir

```powershell
oci os object bulk-upload `
  --bucket-name techmind-models `
  --src-dir .\dist_modelo `
  --object-prefix paquete/v2.0.0/ `
  --content-type auto --overwrite

oci os object list --bucket-name techmind-models --prefix paquete/v2.0.0/ --query "data[].name"
```

Deben aparecer `manifest.json`, los ficheros de `models/`, los de `models/modelo_bertopic/` **y** los de `chroma_db/`.

## 4.5 Crear la Pre-Authenticated Request

Consola → tu bucket → **Pre-Authenticated Requests → Create**:

| Campo | Valor |
|---|---|
| Name | `techmind-paquete-read` |
| PAR Type | **Objects with prefix** |
| Prefix | `paquete/v2.0.0/` |
| Access Type | *Permit object reads on those with the specified prefix* |
| Expiration | una fecha posterior a la presentación |

⚠️ **La URL completa se muestra una sola vez.** Cópiala en ese momento. Es un secreto: quien la tenga puede leer esos objetos.

> Si ya creaste una PAR sobre `models/v2.0.0/` en la ronda anterior, **crea una nueva**: el prefijo ha cambiado a `paquete/`.

## 4.6 El `.env` de la VM

```bash
nano .env
```

```bash
OCI_PAR_URL=https://objectstorage.<region>.oraclecloud.com/p/<token>/n/<namespace>/b/techmind-models/o/paquete/v2.0.0/
OCI_PREFIX=paquete/v2.0.0/
CORS_ORIGINS=http://147.224.239.129
PUBLIC_API_URL=
PRECARGAR_MODELOS=1
```

Dos detalles que rompen la descarga si se te pasan:

- **`OCI_PAR_URL` tiene que terminar en `/`.** El código concatena la ruta relativa detrás. El script de preflight lo comprueba.
- **`PUBLIC_API_URL` tiene que quedar vacía.** Si le pones la IP con puerto, el navegador vuelve a llamar directamente al 8000 y necesitarás abrirlo.

---

# Bloque 5 — Desplegar

## Opción A — construir en la VM (más simple)

```bash
docker compose up --build -d
docker compose logs -f api
```

Espera a `Modelos precargados correctamente`. El primer build tarda: instala torch, hornea el modelo de embeddings y descarga spaCy. Con 2 OCPU, cuenta entre 15 y 30 minutos.

## Opción B — construir en tu PC y publicar en OCIR (más rápido)

Si el build en la VM va lento o se queda sin disco, construye en tu máquina y sube la imagen al **OCI Container Registry**. De paso sumas un segundo servicio de OCI al proyecto, lo cual refuerza el requisito de integración:

```powershell
# <region-key>: iad, phx, gru, scl…   <namespace>: el del bucket
docker login <region-key>.ocir.io -u "<namespace>/<usuario>"
# La contraseña es un Auth Token, NO la de tu cuenta:
# consola → User settings → Auth Tokens → Generate Token

docker build -t <region-key>.ocir.io/<namespace>/techmind-api:2.0.0 .\backend
docker push <region-key>.ocir.io/<namespace>/techmind-api:2.0.0
```

Y en la VM, cambia en `docker-compose.yml`:

```yaml
  api:
    image: <region-key>.ocir.io/<namespace>/techmind-api:2.0.0
    # build: ./backend        <- comentado
```

> Si tu PC es x86_64 y la VM es Ampere, añade `--platform linux/arm64` al `docker build`. Una imagen x86 no arranca en ARM.

---

# Bloque 6 — Verificar

## Desde la VM

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

Con la ronda 2 aplicada, `/health` te dice si el paquete llegó **completo**:

```json
{
  "status": "ok",
  "modelo": {
    "cargado": true,
    "topicos": true,
    "recomendaciones": true,
    "paquete": {"minimos_completos": true, "bertopic": true, "chroma": true}
  }
}
```

Si `topicos` o `recomendaciones` salen `false`, el paquete subió incompleto: vuelve al bloque 4.3.

## Desde tu PC — lo que de verdad cuenta

```powershell
$IP = "147.224.239.129"

curl.exe -i -m 15 http://$IP/
curl.exe -s -m 15 http://$IP/health

curl.exe -s -m 60 -X POST http://$IP/api/v1/contenido `
  -H "Content-Type: application/json" `
  -d '{\"titulo\":\"Introduccion a Spring Boot\",\"texto\":\"En este contenido se presentan los conceptos basicos para la creacion de APIs REST utilizando Java y Spring Boot.\"}'

start http://$IP/analyze
```

**Criterios de aceptación:**

- `categoria` y `probabilidad` poblados
- `informacion_adicional` con palabras clave reales
- `tema.id` **distinto de `-1`**
- `relacionados` **no vacío**
- La web muestra las palabras clave y el tópico en pantalla

## Y lo más importante

Prueba desde **una red distinta a la vuestra** — los datos del móvil sirven. Casi todos los fallos de demo en OCI son de cortafuegos y solo se ven desde fuera.

---

# Bloque 7 — Cuando algo falla

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| `curl` desde fuera se cuelga sin responder | Falta la Security List **o** la regla de `iptables` | Bloque 3, los dos apartados |
| `curl` desde fuera da *Connection refused* | Los contenedores no están arriba | `docker compose ps`, `docker compose logs` |
| El contenedor `api` se reinicia solo | Falta de memoria | `dmesg \| grep -i "killed process"`. Si hay OOM, cambia el shape |
| `/health` devuelve 503 con `modelo_no_disponible` | El paquete no llegó | `docker compose logs api \| grep -i oci`. Revisa que `OCI_PAR_URL` acabe en `/` |
| `/health` responde 200 pero `topicos: false` | Falta `models/modelo_bertopic/` | Reempaqueta con `preparar_paquete_modelo.py` y vuelve a subir |
| `/health` responde 200 pero `recomendaciones: false` | Falta `chroma_db/` | Igual que arriba: el script lo incluye |
| `AttributeError` o `InconsistentVersionWarning` al cargar el `.joblib` | Desajuste de versiones | Falta `requirements.lock.txt` o el Dockerfile sigue en Python 3.10. Bloque 0 |
| El build falla instalando torch | Estás en Ampere con el índice de x86 | Quita `--index-url` del Dockerfile. Bloque 1 |
| `no space left on device` | Disco lleno | `docker system prune -af`. Comprueba `df -h /` |
| El primer análisis tarda muchísimo | Descarga de HuggingFace en caliente | Con la ronda 2 el modelo va horneado en la imagen. Reconstruye |
| La web carga pero el análisis da error de red | `PUBLIC_API_URL` no está vacía | Déjala vacía y reconstruye el frontend |
| `docker compose` dice `permission denied` | Tu usuario no está en el grupo `docker` | `sudo usermod -aG docker $USER` y vuelve a entrar por SSH |

## Comandos de operación

| Acción | Comando |
|---|---|
| Estado | `docker compose ps` |
| Logs de la API | `docker compose logs -f api` |
| Solo lo de OCI | `docker compose logs api \| grep -i oci` |
| Memoria en vivo | `docker stats` |
| Reiniciar la API | `docker compose restart api` |
| Desplegar cambios | `git pull && docker compose up --build -d` |
| Forzar redescarga del paquete | `rm -rf models/* chroma_db/* && docker compose restart api` |

---

# Checklist final

Ejecuta esto la noche anterior, no la mañana de la presentación.

**Antes de tocar la VM**

- [ ] Ronda 2 aplicada y subida a `main`
- [ ] `backend/requirements.lock.txt` existe, con `scikit-learn` igual al del log de entrenamiento
- [ ] `pytest tests\ -q` → 13 passed
- [ ] Actions de GitHub en verde
- [ ] `preparar_paquete_modelo.py` lista `models` (~10 ficheros) **y** `chroma_db`

**OCI**

- [ ] `free -m` ≥ 4 GB (idealmente 12)
- [ ] `df -h /` ≥ 30 GB libres
- [ ] `uname -m` comprobado; si es `aarch64`, Dockerfile ajustado
- [ ] Security List con el 80 abierto
- [ ] `iptables` con el 80 abierto
- [ ] `bash scripts/preflight_oci.sh` sin FALLOs
- [ ] `oci os object list` muestra `manifest.json`, `modelo_bertopic/*` y `chroma_db/*`
- [ ] PAR creada sobre `paquete/v2.0.0/` y pegada en el `.env`, terminada en `/`

**Funcionamiento**

- [ ] `http://<IP>/health` → `topicos: true` y `recomendaciones: true`
- [ ] `POST /api/v1/contenido` con `tema.id != -1` y `relacionados` no vacío
- [ ] `http://<IP>/analyze` muestra palabras clave y tópico
- [ ] `http://<IP>/docs` accesible
- [ ] Probado **desde otra red** (datos del móvil)
- [ ] `docker stats` con memoria estable tras varias peticiones
- [ ] Tres respuestas JSON reales **guardadas en local**, por si la VM falla en vivo
- [ ] El contenedor lleva al menos una hora arrancado

---

## Nota sobre la validación remota

Sigo sin poder alcanzar `147.224.239.129` desde este entorno: el proxy de salida solo permite dominios de su lista y devuelve `403 host_not_allowed` para cualquier IP ajena. Todo lo que he validado hoy —los 13 tests, la sintaxis de `nginx.conf` con `nginx -t`, el estado de `main`, el empaquetado y la descarga del modelo de punta a punta— es sobre el código, no sobre tu despliegue.

Para que pueda decirte qué está pasando de verdad en la VM, pégame la salida de estos tres, ejecutados **desde tu PC**:

```powershell
$IP = "147.224.239.129"
curl.exe -i -m 15 http://$IP/
curl.exe -i -m 15 http://$IP/health
curl.exe -i -m 15 http://$IP:8000/health
```

Y, desde la VM, la de `bash scripts/preflight_oci.sh`.

---

## Fuentes

- [Always Free Resources — Oracle Cloud Infrastructure Documentation](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)
- [OCI Always Free: Updated Ampere A1 Compute Allocation — Oracle Cloud Customer Connect](https://community.oracle.com/customerconnect/discussion/970310/oci-always-free-updated-ampere-a1-compute-allocation)
- [Oracle Quietly Halves Free Tier Ampere A1 Compute Limits — InfoQ](https://www.infoq.com/news/2026/07/oracle-cloud-free-tier-limits/)
