/*
  secretos_ejemplo.h  ->  PLANTILLA
  Haz una copia de este archivo en la misma carpeta, llámala secretos.h
  y escribe ahí los datos reales. secretos.h no se sube a GitHub.

  secretos.h
  Datos privados de la ESP32-CAM. Este archivo debe estar en la MISMA
  carpeta que esp32cam_prueba_deteccion.ino (Arduino IDE lo abre como pestaña).

  NO SUBIR A GITHUB: el repositorio es público.
*/
#pragma once

// Red Wi-Fi a la que se conecta la cámara
const char* WIFI_SSID     = "NOMBRE_DE_LA_RED";
const char* WIFI_PASSWORD = "CONTRASEÑA_DE_LA_RED";

// Clave del dispositivo para /predict.
// Debe ser EXACTAMENTE igual a la variable PREDICT_API_KEY en Render.
const char* API_KEY = "CLAVE_DEL_DISPOSITIVO";
