"""Genera corpus_fallback.csv — corpus de respaldo redactado, sin dependencia de red."""
import csv

D = {}

D["Backend"] = [
("Fundamentos de Spring Boot",
 "Spring Boot es un framework de Java que simplifica la construcción de aplicaciones basadas en Spring mediante autoconfiguración y servidores embebidos. Elimina buena parte del XML de configuración que exigían las versiones anteriores del ecosistema Spring. Un proyecto típico se arranca con Spring Initializr, se declara con anotaciones como RestController y Service, y se empaqueta como un JAR ejecutable que incluye Tomcat integrado."),
("Diseño de APIs REST",
 "Una API REST expone recursos identificados por URIs y manipulados mediante los verbos del protocolo HTTP: GET para lectura, POST para creación, PUT y PATCH para actualización, DELETE para borrado. El estilo arquitectónico REST exige que las interacciones sean sin estado, de modo que cada petición contenga toda la información necesaria para ser procesada. Los códigos de estado HTTP comunican el resultado de la operación al cliente."),
("Arquitectura de microservicios",
 "La arquitectura de microservicios descompone una aplicación monolítica en servicios pequeños y desplegables de forma independiente, cada uno responsable de una capacidad de negocio concreta. Cada servicio administra su propia base de datos y se comunica con los demás mediante APIs sobre HTTP o mensajería asíncrona. El precio de esta independencia es una complejidad operativa mayor: trazabilidad distribuida, consistencia eventual y descubrimiento de servicios."),
("Node.js y programación asíncrona",
 "Node.js es un entorno de ejecución de JavaScript del lado del servidor construido sobre el motor V8 de Chrome. Su modelo de entrada y salida no bloqueante, basado en un bucle de eventos de un solo hilo, lo hace especialmente eficiente para aplicaciones con alta concurrencia de operaciones de red. Express es el framework minimalista más difundido para construir servidores HTTP y APIs sobre Node.js."),
("El framework Django",
 "Django es un framework web de Python que sigue la filosofía de baterías incluidas: trae ORM propio, sistema de migraciones, panel de administración automático y capa de autenticación. Su patrón arquitectónico se conoce como Modelo Vista Plantilla. Django REST Framework extiende el framework con serializadores y vistas orientadas a la construcción de APIs REST sobre los mismos modelos de datos."),
("GraphQL como alternativa a REST",
 "GraphQL es un lenguaje de consulta para APIs que permite al cliente especificar exactamente qué campos necesita, evitando el sobreenvío y el subenvío de datos característicos de los endpoints REST fijos. El servidor expone un único endpoint y un esquema tipado que documenta las operaciones disponibles. Los resolvers son las funciones que obtienen los datos de cada campo del esquema."),
("FastAPI y validación con Pydantic",
 "FastAPI es un framework de Python para construir APIs REST basado en las anotaciones de tipo estándar del lenguaje. Se apoya en Starlette para el manejo asíncrono de peticiones y en Pydantic para la validación y serialización de datos. Genera automáticamente documentación interactiva conforme a la especificación OpenAPI, accesible desde el navegador sin configuración adicional."),
("Middleware en aplicaciones web",
 "Un middleware es un componente que intercepta las peticiones y respuestas que atraviesan una aplicación web, situándose entre el servidor y la lógica de negocio. Se emplea para tareas transversales como el registro de trazas, la autenticación, la compresión de respuestas o la gestión de cabeceras de intercambio de recursos entre orígenes. Los middleware se encadenan en un orden explícito que determina el flujo de ejecución."),
("Colas de mensajes y comunicación asíncrona",
 "Las colas de mensajes desacoplan a los productores de los consumidores de eventos, permitiendo que un servicio publique un mensaje sin conocer quién lo procesará ni cuándo. Sistemas como RabbitMQ o Apache Kafka garantizan la entrega y permiten absorber picos de carga mediante amortiguación. Este patrón es habitual en arquitecturas orientadas a eventos donde el procesamiento no necesita ser inmediato."),
("Caché en el servidor",
 "El almacenamiento en caché reduce la latencia y la carga sobre la base de datos guardando en memoria los resultados de operaciones costosas. Redis y Memcached son las soluciones más difundidas para caché distribuida. La dificultad principal no es escribir en la caché sino invalidarla: definir cuándo un dato almacenado deja de ser válido y debe recalcularse a partir de la fuente de verdad."),
]

D["Frontend"] = [
("La biblioteca React",
 "React es una biblioteca de JavaScript para construir interfaces de usuario mediante componentes reutilizables que encapsulan su propio estado. Utiliza un DOM virtual para calcular el conjunto mínimo de cambios necesarios sobre el árbol real del documento, lo que mejora el rendimiento del renderizado. Los hooks introducidos en la versión dieciséis permiten manejar estado y efectos secundarios en componentes funcionales."),
("Hojas de estilo en cascada",
 "CSS es el lenguaje que describe la presentación visual de un documento HTML, separando el contenido de su apariencia. El modelo de caja define cómo se calculan el ancho, el relleno, el borde y el margen de cada elemento. Los sistemas de disposición modernos, Flexbox y Grid, sustituyeron a las técnicas basadas en flotantes para construir maquetaciones bidimensionales complejas."),
("Diseño web adaptable",
 "El diseño web adaptable permite que una misma página se ajuste a distintos tamaños de pantalla mediante rejillas fluidas, imágenes flexibles y consultas de medios. El enfoque llamado móvil primero parte del diseño para pantallas pequeñas y añade complejidad progresivamente hacia resoluciones mayores. Esta estrategia evita tener que mantener versiones separadas del sitio para cada tipo de dispositivo."),
("TypeScript sobre JavaScript",
 "TypeScript es un superconjunto de JavaScript que añade un sistema de tipos estáticos verificado en tiempo de compilación. El compilador transpila el código a JavaScript estándar ejecutable en cualquier navegador. Los tipos permiten detectar errores antes de la ejecución y habilitan autocompletado y refactorización asistida en el editor, lo cual resulta especialmente valioso en bases de código grandes."),
("El framework Angular",
 "Angular es un framework de desarrollo web mantenido por Google, escrito en TypeScript y basado en una arquitectura de componentes y módulos. Incorpora inyección de dependencias, enrutamiento, formularios reactivos y un cliente HTTP en el propio núcleo del framework. Su enlace de datos bidireccional sincroniza automáticamente el modelo con la vista sin necesidad de código manual de actualización."),
("El Document Object Model",
 "El DOM es la representación en memoria de un documento HTML como un árbol de nodos que los lenguajes de script pueden consultar y modificar. Cada elemento de la página es un nodo con propiedades, atributos y descendientes. Manipular el DOM directamente es costoso en términos de rendimiento, razón por la cual las bibliotecas modernas introducen capas de abstracción que agrupan y minimizan esas operaciones."),
("Aplicaciones web progresivas",
 "Una aplicación web progresiva combina las capacidades de un sitio web con la experiencia de una aplicación nativa: puede instalarse en el dispositivo, funcionar sin conexión y recibir notificaciones push. El service worker es el componente central, un script que se ejecuta en segundo plano e intercepta las peticiones de red para servirlas desde una caché local cuando no hay conectividad."),
("Empaquetadores de módulos",
 "Los empaquetadores como Webpack o Vite resuelven el grafo de dependencias de una aplicación y producen paquetes optimizados para el navegador. Aplican transformaciones como la transpilación de sintaxis moderna, la eliminación de código no utilizado y la división del paquete en fragmentos cargados bajo demanda. Vite gana terreno por su servidor de desarrollo basado en módulos nativos del navegador."),
("Gestión de estado en el cliente",
 "A medida que una interfaz crece, mantener el estado sincronizado entre componentes distantes del árbol se vuelve difícil. Las bibliotecas de gestión de estado centralizan los datos compartidos en un almacén único y definen un flujo de actualización unidireccional y predecible. Redux popularizó este patrón con acciones y reductores; alternativas más recientes reducen considerablemente el código repetitivo necesario."),
("Accesibilidad web",
 "La accesibilidad web consiste en diseñar sitios utilizables por personas con discapacidades visuales, auditivas o motoras. Implica usar etiquetas HTML semánticas, proporcionar texto alternativo para las imágenes, garantizar contraste suficiente y asegurar que toda la funcionalidad sea operable mediante teclado. Los atributos del estándar ARIA describen el rol y el estado de componentes interactivos para los lectores de pantalla."),
]

D["Data Science"] = [
("Aprendizaje automático supervisado",
 "El aprendizaje supervisado entrena un modelo a partir de ejemplos etiquetados, ajustando sus parámetros para minimizar el error entre las predicciones y las etiquetas reales. Los problemas se dividen en clasificación, cuando la salida es una categoría discreta, y regresión, cuando la salida es un valor continuo. La generalización a datos no vistos es el objetivo real, no el ajuste al conjunto de entrenamiento."),
("Regresión logística",
 "La regresión logística es un modelo lineal de clasificación que estima la probabilidad de pertenencia a una clase aplicando la función sigmoide a una combinación lineal de las variables de entrada. A pesar de su simplicidad sigue siendo una referencia sólida en clasificación de texto, donde las representaciones dispersas de alta dimensión favorecen a los modelos lineales frente a alternativas más complejas."),
("La representación TF-IDF",
 "TF-IDF pondera cada término de un documento multiplicando su frecuencia local por el logaritmo inverso de la proporción de documentos del corpus que lo contienen. Un término que aparece en casi todos los documentos recibe un peso bajo porque aporta poca capacidad discriminante. El resultado es una matriz dispersa donde cada fila representa un documento en el espacio del vocabulario."),
("Agrupamiento con K-medias",
 "El algoritmo de K-medias particiona un conjunto de observaciones en k grupos minimizando la suma de distancias cuadradas de cada punto al centroide de su grupo. Requiere fijar k de antemano y asume grupos de forma aproximadamente esférica y tamaño similar. El método del codo y el coeficiente de silueta son las heurísticas habituales para elegir un valor razonable de k."),
("Procesamiento de lenguaje natural",
 "El procesamiento de lenguaje natural abarca las técnicas que permiten a una máquina analizar y generar lenguaje humano. Las etapas clásicas de un pipeline incluyen la tokenización, la eliminación de palabras vacías, la lematización y el etiquetado gramatical. Los modelos basados en la arquitectura transformador desplazaron a los enfoques estadísticos previos en la mayoría de las tareas del área."),
("Embeddings semánticos de texto",
 "Un embedding representa un fragmento de texto como un vector denso de dimensión fija en el que la proximidad geométrica refleja similitud de significado. A diferencia de las representaciones basadas en conteo de palabras, dos textos que expresan la misma idea con vocabulario distinto producen vectores cercanos. Esta propiedad habilita la búsqueda semántica y la recomendación de contenido relacionado."),
("Validación cruzada",
 "La validación cruzada evalúa la capacidad de generalización de un modelo dividiendo repetidamente los datos en subconjuntos de entrenamiento y validación. En la variante de k particiones, el conjunto se divide en k bloques y el modelo se entrena k veces dejando fuera uno distinto cada vez. La estratificación preserva la proporción de clases en cada partición, algo necesario en conjuntos desbalanceados."),
("Métricas de clasificación",
 "La exactitud resulta engañosa en conjuntos desbalanceados porque una clase mayoritaria puede dominar el resultado. La precisión mide qué proporción de las predicciones positivas era correcta y la exhaustividad qué proporción de los positivos reales fue recuperada. La medida F1 es su media armónica, y su promedio macro pondera todas las clases por igual con independencia de su frecuencia."),
("Reducción de dimensionalidad",
 "Las técnicas de reducción de dimensionalidad proyectan datos de alta dimensión a un espacio menor conservando la mayor cantidad posible de estructura. El análisis de componentes principales busca las direcciones ortogonales de máxima varianza. Métodos no lineales como UMAP preservan mejor la estructura local de vecindad y se usan habitualmente para visualizar agrupamientos en dos dimensiones."),
("Sobreajuste y regularización",
 "Un modelo sobreajustado memoriza el ruido del conjunto de entrenamiento y falla al generalizar sobre datos nuevos. La señal característica es una brecha amplia entre el desempeño en entrenamiento y en validación. La regularización penaliza la magnitud de los coeficientes del modelo para restringir su complejidad, siendo las variantes L1 y L2 las formulaciones más habituales."),
]

D["DevOps"] = [
("Contenedores con Docker",
 "Docker empaqueta una aplicación junto con todas sus dependencias en una imagen inmutable que se ejecuta de forma idéntica en cualquier entorno. A diferencia de una máquina virtual, un contenedor comparte el núcleo del sistema operativo anfitrión, por lo que arranca en segundos y consume muchos menos recursos. El Dockerfile describe de forma declarativa y versionable cómo se construye la imagen."),
("Orquestación con Kubernetes",
 "Kubernetes automatiza el despliegue, el escalado y la operación de aplicaciones en contenedores sobre un clúster de máquinas. Su unidad mínima de despliegue es el pod, que agrupa uno o más contenedores que comparten red y almacenamiento. El estado deseado se declara en manifiestos y el plano de control reconcilia continuamente el estado real del clúster con esa declaración."),
("Integración continua",
 "La integración continua es la práctica de fusionar frecuentemente los cambios de todos los desarrolladores en una rama compartida, validando cada fusión con una construcción y una batería de pruebas automáticas. Detectar los conflictos de integración a diario, en lugar de acumularlos durante semanas, reduce drásticamente el coste de resolverlos y mantiene la rama principal siempre desplegable."),
("Entrega y despliegue continuos",
 "La entrega continua extiende la integración continua garantizando que cada cambio que supera el pipeline queda listo para publicarse en producción. El despliegue continuo va un paso más allá y automatiza también la publicación, sin intervención humana. Las estrategias de despliegue azul verde o canario permiten liberar cambios de forma gradual y revertirlos rápidamente ante un fallo."),
("Control de versiones con Git",
 "Git es un sistema de control de versiones distribuido en el que cada copia del repositorio contiene el historial completo del proyecto. Las ramas son punteros ligeros a confirmaciones, lo que hace que crear y fusionar ramas sea una operación barata. Los flujos de trabajo basados en ramas de funcionalidad y revisión mediante solicitudes de fusión son el estándar en equipos de desarrollo."),
("Infraestructura como código",
 "La infraestructura como código gestiona servidores, redes y servicios mediante archivos de configuración versionados en lugar de configuración manual. Herramientas declarativas como Terraform describen el estado deseado de la infraestructura y calculan el plan de cambios necesario para alcanzarlo. Esto hace que los entornos sean reproducibles y elimina la deriva de configuración entre desarrollo y producción."),
("Observabilidad de sistemas",
 "La observabilidad se apoya en tres pilares complementarios: las métricas, que cuantifican el comportamiento agregado del sistema; los registros, que documentan eventos discretos; y las trazas distribuidas, que siguen una petición a través de todos los servicios que atraviesa. En una arquitectura distribuida, la traza es a menudo el único modo de localizar el origen real de una latencia elevada."),
("Automatización con pipelines",
 "Un pipeline de despliegue encadena etapas de construcción, prueba, análisis estático y publicación, deteniéndose ante el primer fallo. Definirlo como código en el propio repositorio permite versionarlo y revisarlo junto con la aplicación. Plataformas como GitHub Actions, GitLab CI o Jenkins ejecutan estas etapas en entornos efímeros y aislados para garantizar la reproducibilidad."),
("Gestión de configuración y secretos",
 "Las credenciales, claves de API y cadenas de conexión nunca deben incluirse en el código fuente. El enfoque habitual es inyectarlas como variables de entorno en tiempo de ejecución y almacenarlas cifradas en un gestor de secretos dedicado. Rotar periódicamente las credenciales y aplicar el principio de mínimo privilegio limita el impacto de una filtración accidental."),
("Escalado horizontal y balanceo de carga",
 "El escalado horizontal añade instancias de una aplicación en lugar de aumentar los recursos de una sola máquina. Un balanceador de carga distribuye las peticiones entrantes entre las instancias disponibles y retira del reparto aquellas que fallan las comprobaciones de salud. Este modelo exige que la aplicación sea sin estado, delegando la sesión a un almacén externo compartido."),
]

D["Bases de Datos"] = [
("El sistema PostgreSQL",
 "PostgreSQL es un sistema de gestión de bases de datos relacional de código abierto con soporte transaccional completo y control de concurrencia multiversión. Ofrece tipos de datos avanzados como JSONB y arreglos nativos, además de un mecanismo de extensiones que amplía sus capacidades. La extensión pgvector añade indexación y búsqueda por similitud sobre vectores de alta dimensión."),
("El lenguaje SQL",
 "SQL es el lenguaje estándar para consultar y manipular bases de datos relacionales. Sus cláusulas fundamentales permiten seleccionar columnas, filtrar filas mediante condiciones, combinar tablas relacionadas, agrupar resultados y calcular agregaciones. Las combinaciones internas y externas determinan qué ocurre con las filas que no encuentran correspondencia en la tabla contraria."),
("Normalización de bases de datos",
 "La normalización organiza las tablas de una base de datos relacional para reducir la redundancia y evitar anomalías de inserción, actualización y borrado. Las tres primeras formas normales eliminan progresivamente los grupos repetitivos y las dependencias funcionales parciales y transitivas. En sistemas orientados a lectura intensiva se practica a veces una desnormalización controlada por razones de rendimiento."),
("Índices y planes de ejecución",
 "Un índice es una estructura auxiliar, habitualmente un árbol B, que acelera la localización de filas evitando el recorrido completo de la tabla. El coste es un mayor consumo de espacio y una penalización en las operaciones de escritura, que deben mantener el índice actualizado. El planificador de consultas decide si usarlo comparando el coste estimado de las alternativas disponibles."),
("Transacciones y propiedades ACID",
 "Una transacción agrupa varias operaciones en una unidad atómica que se confirma completa o se revierte por entero. Las propiedades ACID garantizan atomicidad, consistencia, aislamiento y durabilidad. Los niveles de aislamiento definen qué fenómenos concurrentes se permiten, desde lecturas sucias hasta lecturas fantasma, equilibrando corrección y rendimiento bajo concurrencia."),
("Bases de datos NoSQL",
 "Las bases de datos NoSQL renuncian a parte del modelo relacional para ganar escalabilidad horizontal y flexibilidad de esquema. Se agrupan en familias según su modelo de datos: documentales, de clave valor, de familias de columnas y de grafos. El teorema CAP formaliza que un sistema distribuido no puede garantizar simultáneamente consistencia, disponibilidad y tolerancia a particiones."),
("MongoDB y el modelo documental",
 "MongoDB almacena los datos como documentos con estructura similar a JSON agrupados en colecciones, sin exigir un esquema fijo previo. Este modelo encaja bien cuando los registros tienen estructura variable o cuando conviene mantener juntos datos que se consultan siempre a la vez. Las agregaciones se expresan como una tubería de etapas que transforman progresivamente los documentos."),
("Redis como almacén en memoria",
 "Redis es un almacén de estructuras de datos en memoria que se emplea como caché, intermediario de mensajes y base de datos de baja latencia. Soporta cadenas, listas, conjuntos, conjuntos ordenados y tablas hash como tipos nativos. La persistencia opcional en disco mediante instantáneas o registro de operaciones permite recuperar el estado tras un reinicio del proceso."),
("Bases de datos vectoriales",
 "Una base de datos vectorial indexa vectores de alta dimensión y responde consultas de vecinos más cercanos de forma eficiente, algo que las estructuras de índice relacionales tradicionales no resuelven. Utiliza estructuras aproximadas como los grafos jerárquicos navegables de mundo pequeño. Almacena metadatos junto a cada vector, lo que permite combinar filtrado estructurado y búsqueda por similitud."),
("Migraciones de esquema",
 "Una migración es un cambio versionado en la estructura de la base de datos, expresado como código y aplicado de forma ordenada y reproducible en todos los entornos. Herramientas como Alembic o Flyway registran qué migraciones se aplicaron y permiten avanzar o revertir el esquema. Las migraciones destructivas requieren estrategias de despliegue en varias fases para no interrumpir el servicio."),
]

D["Cloud"] = [
("Computación en la nube",
 "La computación en la nube ofrece recursos de cómputo, almacenamiento y red bajo demanda a través de internet, con pago por consumo real. Sustituye la inversión inicial en hardware por un gasto operativo variable y permite aprovisionar capacidad en minutos. La elasticidad, es decir la capacidad de crecer y decrecer según la carga, es su ventaja diferencial frente al centro de datos propio."),
("Modelos de servicio en la nube",
 "La infraestructura como servicio entrega recursos virtualizados básicos como máquinas y discos, dejando al cliente la gestión del sistema operativo. La plataforma como servicio abstrae también el entorno de ejecución, de modo que el equipo solo despliega código. El software como servicio entrega la aplicación terminada, sin que el usuario administre ninguna capa de infraestructura."),
("Almacenamiento de objetos",
 "El almacenamiento de objetos guarda los datos como unidades independientes con metadatos asociados dentro de contenedores planos llamados buckets, en lugar de una jerarquía de directorios. Está diseñado para durabilidad extremadamente alta y acceso mediante API HTTP. Es la opción habitual para archivos estáticos, copias de seguridad y artefactos de modelos de aprendizaje automático."),
("Computación sin servidor",
 "El modelo sin servidor ejecuta funciones en respuesta a eventos sin que el desarrollador aprovisione ni administre servidores. El proveedor asigna recursos automáticamente y factura por tiempo de ejecución y memoria consumida. Sus limitaciones principales son la latencia del arranque en frío y los tiempos máximos de ejecución, que lo hacen inadecuado para procesos largos y continuos."),
("Oracle Cloud Infrastructure",
 "Oracle Cloud Infrastructure ofrece un catálogo de servicios que incluye máquinas virtuales, almacenamiento de objetos, bases de datos autónomas, funciones sin servidor y una pasarela de API. Los recursos se organizan en compartimentos que delimitan el ámbito de las políticas de acceso. El servicio Vault gestiona claves de cifrado y secretos de forma centralizada."),
("Redes virtuales en la nube",
 "Una red virtual en la nube aísla lógicamente los recursos de un cliente dentro de la infraestructura compartida del proveedor. Se divide en subredes públicas y privadas, y el tráfico entre ellas se controla mediante listas de seguridad y tablas de enrutamiento. Los recursos que no requieren exposición directa a internet deben ubicarse siempre en subredes privadas."),
("Alta disponibilidad y regiones",
 "Los proveedores de nube organizan su infraestructura en regiones geográficas divididas en zonas de disponibilidad con alimentación y red independientes. Distribuir las instancias de una aplicación entre varias zonas la protege frente al fallo de una instalación completa. La replicación entre regiones distintas añade resistencia ante desastres a costa de latencia y coste de transferencia."),
("Costes y optimización en la nube",
 "El modelo de pago por uso traslada el control del gasto al diseño de la arquitectura. Las principales fuentes de coste inesperado son las instancias sobredimensionadas, el almacenamiento no utilizado y la transferencia de datos hacia fuera del proveedor. Las etiquetas de recursos permiten atribuir el gasto por equipo o proyecto y detectar desviaciones antes de que se consoliden."),
("Nube híbrida y multinube",
 "Una arquitectura híbrida combina infraestructura propia con servicios de nube pública, habitualmente por requisitos regulatorios o por inversiones previas en hardware. La estrategia multinube distribuye las cargas entre varios proveedores para reducir la dependencia de uno solo. Ambos enfoques aumentan la complejidad de la red, la identidad y la observabilidad de forma considerable."),
("Contenedores gestionados en la nube",
 "Los servicios gestionados de contenedores operan el plano de control de Kubernetes por cuenta del cliente, que solo administra los nodos de trabajo y sus cargas. Esto elimina la tarea de mantener actualizado y disponible el propio orquestador. La integración con el balanceador de carga, el almacenamiento persistente y la gestión de identidades del proveedor viene resuelta de fábrica."),
]

D["Mobile"] = [
("Desarrollo en Android",
 "Android es un sistema operativo móvil basado en el núcleo Linux cuyas aplicaciones se desarrollan principalmente en Kotlin o Java. Los componentes fundamentales de una aplicación son las actividades, los servicios, los receptores de anuncios y los proveedores de contenido. El ciclo de vida de una actividad determina en qué momento la aplicación debe liberar o restaurar recursos y estado."),
("El lenguaje Kotlin",
 "Kotlin es un lenguaje de programación que se ejecuta sobre la máquina virtual de Java y es plenamente interoperable con el código Java existente. Su sistema de tipos distingue las referencias que admiten nulo de las que no, eliminando en compilación una clase entera de errores en tiempo de ejecución. Las corrutinas ofrecen un modelo de concurrencia ligero y legible para operaciones asíncronas."),
("Desarrollo en iOS",
 "Las aplicaciones para iOS se desarrollan en Swift u Objective-C usando el entorno Xcode y los marcos de trabajo de Apple. SwiftUI introdujo un modelo declarativo de construcción de interfaces que reemplaza progresivamente al enfoque imperativo basado en UIKit. La distribución pasa obligatoriamente por el proceso de revisión de la App Store, que impone requisitos estrictos de privacidad."),
("Flutter y el desarrollo multiplataforma",
 "Flutter es un kit de desarrollo de Google que permite construir aplicaciones para móvil, escritorio y web desde una única base de código escrita en Dart. En lugar de usar los componentes nativos de cada plataforma, dibuja su propia interfaz mediante un motor de renderizado propio, lo que garantiza apariencia idéntica en todos los dispositivos a costa de cierto peso adicional."),
("React Native",
 "React Native permite escribir aplicaciones móviles usando JavaScript y el modelo de componentes de React, traduciendo esos componentes a widgets nativos de cada plataforma. Comparte gran parte de la lógica entre Android e iOS mientras conserva el aspecto propio de cada sistema. Las funcionalidades que requieren capacidades específicas del dispositivo se implementan mediante módulos nativos."),
("Arquitectura de aplicaciones móviles",
 "Los patrones arquitectónicos móviles separan la lógica de presentación del modelo de datos para facilitar las pruebas y el mantenimiento. Modelo Vista Vista Modelo es el más difundido, con una capa intermedia que expone al vista el estado ya preparado para mostrarse. Un repositorio centraliza el acceso a datos y decide entre la caché local y la fuente remota."),
("Persistencia local en el dispositivo",
 "Las aplicaciones móviles necesitan almacenar datos localmente para funcionar sin conexión y reducir la latencia percibida. SQLite es el motor embebido estándar en ambas plataformas, habitualmente accedido mediante una capa de abstracción. La sincronización posterior con el servidor debe resolver los conflictos que surgen cuando el mismo registro se modificó en ambos extremos."),
("Notificaciones push",
 "Las notificaciones push permiten al servidor iniciar la comunicación con la aplicación aunque esta no esté en ejecución. El dispositivo se registra ante el servicio de mensajería de la plataforma y obtiene un identificador que el servidor almacena. El abuso de esta capacidad es una causa frecuente de desinstalación, por lo que la segmentación y la frecuencia requieren criterio."),
("Rendimiento y consumo de batería",
 "El rendimiento percibido en móvil depende críticamente del tiempo de arranque y de la fluidez del desplazamiento. Las operaciones de red y de disco deben ejecutarse siempre fuera del hilo principal para no bloquear la interfaz. El consumo de batería se dispara con la geolocalización continua, el sondeo periódico del servidor y los procesos que impiden la suspensión del dispositivo."),
("Publicación y distribución de aplicaciones",
 "Publicar una aplicación exige firmar el paquete con un certificado, cumplir las políticas de la tienda y declarar los permisos y prácticas de privacidad. Los canales de prueba internos y abiertos permiten distribuir versiones preliminares a grupos limitados antes del lanzamiento general. El despliegue escalonado libera la actualización a un porcentaje creciente de usuarios."),
]

D["Seguridad"] = [
("Autenticación y autorización",
 "La autenticación verifica la identidad de quien realiza una petición, mientras que la autorización determina qué acciones tiene permitido ejecutar esa identidad ya verificada. Confundir ambos conceptos es una fuente habitual de vulnerabilidades. El control de acceso basado en roles agrupa los permisos en perfiles que se asignan a los usuarios en lugar de concederlos individualmente."),
("El protocolo OAuth",
 "OAuth es un marco de autorización que permite a una aplicación acceder a recursos de un usuario en otro servicio sin conocer sus credenciales. El usuario autoriza explícitamente el acceso y la aplicación recibe un testigo de alcance y duración limitados. El flujo de código de autorización con clave de prueba es el recomendado para aplicaciones públicas que no pueden guardar un secreto."),
("Testigos web JSON",
 "Un JSON Web Token es una cadena firmada que transporta afirmaciones sobre una identidad y puede validarse sin consultar una base de datos. Consta de cabecera, cuerpo y firma codificados en base64url. Su contenido es legible por cualquiera, por lo que no debe incluir datos sensibles, y su revocación anticipada exige mecanismos adicionales como listas de invalidación."),
("Inyección SQL",
 "La inyección SQL ocurre cuando una entrada del usuario se concatena directamente en una consulta, permitiendo alterar su estructura y ejecutar operaciones no previstas. La defensa efectiva son las consultas parametrizadas, que separan el código SQL de los datos y hacen imposible que la entrada modifique la sentencia. Los mapeadores objeto relacional aplican esta parametrización por defecto."),
("Cifrado en tránsito y en reposo",
 "El cifrado en tránsito protege los datos mientras viajan por la red, y el protocolo TLS es el estándar que lo implementa sobre HTTP. El cifrado en reposo protege la información almacenada en discos y copias de seguridad frente al acceso físico o lógico no autorizado. La gestión del ciclo de vida de las claves suele ser el eslabón más débil de ambos esquemas."),
("Funciones de derivación de contraseñas",
 "Las contraseñas nunca deben almacenarse en texto plano ni con funciones de resumen rápidas de propósito general. Los algoritmos diseñados para este fin, como bcrypt, scrypt o Argon2, son deliberadamente costosos en tiempo y memoria para encarecer los ataques por fuerza bruta. La sal aleatoria por usuario impide el uso de tablas precalculadas de resúmenes."),
("Vulnerabilidades web comunes",
 "El proyecto OWASP publica periódicamente una lista de los riesgos más críticos en aplicaciones web. Entre los recurrentes figuran el control de acceso defectuoso, los fallos criptográficos, la inyección y la configuración de seguridad incorrecta. La ejecución de scripts entre sitios permite inyectar código en el navegador de otros usuarios y se mitiga escapando toda salida no confiable."),
("Cortafuegos y segmentación de red",
 "Un cortafuegos filtra el tráfico de red aplicando reglas basadas en direcciones, puertos y protocolos, y constituye la primera línea de defensa perimetral. La segmentación divide la red en zonas con distinto nivel de confianza para limitar el movimiento lateral de un atacante que ya obtuvo acceso. El principio de denegar por defecto debe regir toda configuración de reglas."),
("Gestión de dependencias vulnerables",
 "Una proporción elevada del código de una aplicación moderna proviene de dependencias de terceros que el equipo no escribió ni audita. Los análisis automáticos de composición de software detectan bibliotecas con vulnerabilidades publicadas y proponen versiones corregidas. Fijar las versiones exactas y verificar la integridad de los paquetes reduce el riesgo de ataques a la cadena de suministro."),
("Registro de auditoría y respuesta a incidentes",
 "Un registro de auditoría documenta quién hizo qué y cuándo sobre los recursos sensibles del sistema, y debe almacenarse de forma inmutable y separada de la aplicación que lo genera. Sin estos registros, reconstruir el alcance real de una brecha resulta imposible. Un plan de respuesta a incidentes define de antemano los roles, la comunicación y los pasos de contención."),
]

filas = []
i = 0
for categoria, docs in D.items():
    for titulo, texto in docs:
        filas.append({
            "doc_id": f"DOC-{i:04d}",
            "titulo": titulo,
            "texto": texto,
            "texto_limpio": texto,
            "categoria": categoria,
            "n_chars": len(texto),
            "fuente": "corpus_fallback (redactado por el equipo)",
        })
        i += 1

with open("corpus_fallback.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["doc_id", "titulo", "texto", "texto_limpio",
                                      "categoria", "n_chars", "fuente"])
    w.writeheader()
    w.writerows(filas)

print(f"{len(filas)} documentos | {len(D)} categorías")
print(f"longitud: min={min(r['n_chars'] for r in filas)} "
      f"media={sum(r['n_chars'] for r in filas)/len(filas):.0f} "
      f"max={max(r['n_chars'] for r in filas)}")
for c, d in D.items():
    print(f"  {c:<16} {len(d)}")
