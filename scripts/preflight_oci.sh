#!/usr/bin/env bash
# Comprobacion previa al despliegue de TechMind en una VM de OCI.
#
# Uso, dentro de la instancia:
#     bash scripts/preflight_oci.sh
#
# No modifica nada. Solo mira y avisa. Ejecutalo ANTES de `docker compose up`:
# cada uno de estos controles corresponde a un fallo de despliegue real.

set -uo pipefail

OK=0; AVISOS=0; FALLOS=0
USUARIO="${USER:-$(id -un)}"

verde()  { printf '  \033[32m[ OK   ]\033[0m %s\n' "$1"; OK=$((OK+1)); }
ambar()  { printf '  \033[33m[ AVISO]\033[0m %s\n' "$1"; AVISOS=$((AVISOS+1)); }
rojo()   { printf '  \033[31m[ FALLO]\033[0m %s\n' "$1"; FALLOS=$((FALLOS+1)); }
titulo() { printf '\n\033[1m%s\033[0m\n%s\n' "$1" "$(printf '%.0s-' {1..62})"; }

printf '\n==============================================================\n'
printf '  TechMind — comprobacion previa al despliegue en OCI\n'
printf '==============================================================\n'

# ------------------------------------------------------------------ #
titulo "1. Recursos de la instancia"

RAM_MB=$(free -m | awk '/^Mem:/{print $2}')
if   [ "$RAM_MB" -lt 2000 ]; then
    rojo "RAM: ${RAM_MB} MB. Insuficiente — torch + sentence-transformers + BERTopic no caben."
    echo "           El contenedor morira por OOM. Cambia el shape a Ampere A1 (2 OCPU / 12 GB)."
elif [ "$RAM_MB" -lt 4000 ]; then
    ambar "RAM: ${RAM_MB} MB. Justo. Puede arrancar, pero el margen es minimo."
else
    verde "RAM: ${RAM_MB} MB"
fi

DISCO_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if   [ "$DISCO_GB" -lt 15 ]; then
    rojo "Disco libre: ${DISCO_GB} GB. La imagen con torch no cabe (necesita ~10-15 GB)."
elif [ "$DISCO_GB" -lt 30 ]; then
    ambar "Disco libre: ${DISCO_GB} GB. Suficiente para un despliegue, escaso para reconstruir."
else
    verde "Disco libre: ${DISCO_GB} GB"
fi

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    ambar "Arquitectura: aarch64 (Ampere ARM)."
    echo "           En ARM, QUITA '--index-url https://download.pytorch.org/whl/cpu'"
    echo "           de backend/Dockerfile: ese indice esta pensado para el reparto"
    echo "           CUDA/CPU de x86_64. En ARM el torch de PyPI ya es solo-CPU."
else
    verde "Arquitectura: ${ARCH}"
fi

# ------------------------------------------------------------------ #
titulo "2. Docker"

if command -v docker >/dev/null 2>&1; then
    verde "docker instalado ($(docker --version | cut -d, -f1))"
    if docker info >/dev/null 2>&1; then
        verde "el usuario '${USUARIO}' puede hablar con el daemon"
    else
        rojo "el daemon no responde para '${USUARIO}'."
        echo "           sudo systemctl enable --now docker"
        echo "           sudo usermod -aG docker \$USER   # y vuelve a entrar por SSH"
    fi
    docker compose version >/dev/null 2>&1 \
        && verde "plugin docker compose presente" \
        || rojo "falta el plugin compose: instala docker-compose-plugin"
else
    rojo "docker no esta instalado"
fi

# ------------------------------------------------------------------ #
titulo "3. Cortafuegos del sistema operativo"

# Son DOS cortafuegos independientes: este y la Security List de la consola
# de OCI. Este script solo puede ver el del sistema operativo.
ABIERTO_80=no
if command -v firewall-cmd >/dev/null 2>&1 && firewall-cmd --state >/dev/null 2>&1; then
    firewall-cmd --list-ports 2>/dev/null | grep -q '80/tcp' && ABIERTO_80=si
fi
if sudo -n iptables -S INPUT 2>/dev/null | grep -qE '\-\-dport 80 .*ACCEPT'; then
    ABIERTO_80=si
fi

if [ "$ABIERTO_80" = "si" ]; then
    verde "puerto 80 permitido en el cortafuegos del sistema operativo"
else
    ambar "no veo una regla para el puerto 80 (puede que no tenga permisos para mirar)."
    echo "           sudo firewall-cmd --permanent --add-port=80/tcp && sudo firewall-cmd --reload"
    echo "           sudo iptables -I INPUT 6 -p tcp --dport 80 -j ACCEPT"
fi
echo "           Recuerda: esto NO comprueba la Security List de la consola de OCI."

# ------------------------------------------------------------------ #
titulo "4. Configuracion del proyecto"

[ -f .env ] && verde "existe .env" \
            || rojo "falta .env — copialo de .env.example y rellenalo"

if [ -f .env ]; then
    # shellcheck disable=SC1091
    set -a; . ./.env 2>/dev/null; set +a

    if [ -n "${OCI_PAR_URL:-}" ]; then
        verde "OCI_PAR_URL definida"
        case "${OCI_PAR_URL}" in
            */) verde "OCI_PAR_URL termina en '/' (correcto)" ;;
            *)  rojo "OCI_PAR_URL NO termina en '/': las rutas se concatenaran mal" ;;
        esac
    elif [ -n "${OCI_NAMESPACE:-}" ] && [ -n "${OCI_BUCKET:-}" ]; then
        verde "OCI_NAMESPACE y OCI_BUCKET definidos (modo SDK)"
    else
        ambar "sin OCI_PAR_URL ni OCI_NAMESPACE/OCI_BUCKET."
        echo "           Solo funcionara si models/ y chroma_db/ ya estan en disco."
    fi

    [ -n "${PUBLIC_API_URL:-}" ] \
        && ambar "PUBLIC_API_URL='${PUBLIC_API_URL}'. Dejala VACIA para usar el proxy de nginx." \
        || verde "PUBLIC_API_URL vacia (rutas relativas via nginx)"
fi

# ------------------------------------------------------------------ #
titulo "5. Paquete del modelo en disco"

faltan=0
for f in metadata.json modelo_clasificacion.joblib label_encoder.joblib \
         config.json centroides_clase.joblib tecnologias.json; do
    [ -f "models/$f" ] || { faltan=$((faltan+1)); }
done

if [ "$faltan" -eq 0 ]; then
    verde "los 6 artefactos minimos estan en models/"
else
    ambar "faltan $faltan de los 6 artefactos minimos en models/."
    echo "           Se descargaran de Object Storage al arrancar (si hay PAR configurada)."
fi

[ -d models/modelo_bertopic ] \
    && verde "models/modelo_bertopic presente — el campo 'tema' funcionara" \
    || ambar "falta models/modelo_bertopic — el campo 'tema' vendra VACIO"

[ -d chroma_db ] && [ -n "$(ls -A chroma_db 2>/dev/null)" ] \
    && verde "chroma_db poblada — las recomendaciones funcionaran" \
    || ambar "chroma_db vacia — el campo 'relacionados' vendra VACIO"

# ------------------------------------------------------------------ #
titulo "6. Salida a internet"

curl -fsS -m 10 -o /dev/null https://huggingface.co 2>/dev/null \
    && verde "huggingface.co accesible" \
    || ambar "no llego a huggingface.co. Si el modelo de embeddings no esta"$'\n'"           horneado en la imagen, el arranque fallara."

curl -fsS -m 10 -o /dev/null https://registry-1.docker.io/v2/ 2>/dev/null \
    && verde "registro de Docker accesible" \
    || ambar "no llego al registro de Docker: 'docker compose build' fallara"

# ------------------------------------------------------------------ #
printf '\n==============================================================\n'
printf '  RESUMEN:  %d correctos  ·  %d avisos  ·  %d fallos\n' "$OK" "$AVISOS" "$FALLOS"
printf '==============================================================\n'
if [ "$FALLOS" -gt 0 ]; then
    printf '  Resuelve los FALLOS antes de ejecutar docker compose up.\n\n'
    exit 1
fi
printf '  Listo para desplegar.\n\n'
