# Código de la ESP32-CAM

- `esp32cam_envio_yolo`: versión normal. Una foto por minuto, solo en el horario de la sede y sin festivos.
- `esp32cam_prueba_deteccion`: versión de prueba. Una foto cada 30 segundos, a cualquier hora. Solo para ajustar la cámara o medir la precisión.

## Antes de cargarlo

En la carpeta de la versión que vas a usar, copia `secretos_ejemplo.h` como `secretos.h` y escribe:

- el nombre y la contraseña de la red Wi-Fi (2,4 GHz);
- la clave del dispositivo, que debe ser igual a la variable `PREDICT_API_KEY` del servidor en Render (la entrega el administrador).

`secretos.h` no se sube al repositorio, porque es público.

## Configuración en Arduino IDE

- Placa: AI Thinker ESP32-CAM
- PSRAM: Enabled
- Partition Scheme: Huge APP (3MB No OTA/1MB SPIFFS)
- Monitor Serie: 115200 baudios

En la versión normal, cambia `ORIGEN` por el módulo donde quedará la cámara (por ejemplo `"modulo-5"`) y activa "Con cámara" para ese módulo en el panel del administrador.
