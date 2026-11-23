/*
  esp32cam_prueba_deteccion.ino  (SOLO PARA PRUEBAS)
  ESP32-CAM AI Thinker (sensor OV3660)

  Versión de prueba del código de la cámara. Sirve para revisar la detección
  mientras ajustan la posición de la cámara o hacen las pruebas de precisión.

  Diferencias con el código normal (esp32cam_envio_yolo):
    - Toma y envía una foto cada 30 segundos (el normal, cada 60).
    - Funciona a cualquier hora y cualquier día, también de noche, domingos y
      festivos: no revisa el horario de atención y nunca se duerme.

  IMPORTANTE:
    - No dejarla instalada en el módulo: gasta el doble de datos del hotspot y
      mantiene despierto el servidor de Render, que tiene horas gratis limitadas.
    - Las lecturas de prueba se guardan como cualquier otra y aparecen en la
      página y en las estadísticas.
    - Al terminar las pruebas, volver a subir esp32cam_envio_yolo.

  Se carga igual que el normal: abrir este .ino en Arduino IDE (secretos.h debe
  estar en la misma carpeta) con la misma configuración de placa:
    Placa:               AI Thinker ESP32-CAM
    PSRAM:               Enabled
    Partition Scheme:    Huge APP (3MB No OTA/1MB SPIFFS)
    Monitor Serie:       115200 baudios
*/

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <time.h>
#include "esp_camera.h"
#include "esp_sleep.h"
#include "driver/rtc_io.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// Wi-Fi (WIFI_SSID, WIFI_PASSWORD) y clave del dispositivo (API_KEY)
#include "secretos.h"

// Declaración anticipada (la función está más abajo)
String leerLinea(WiFiClientSecure& cliente, unsigned long limite);

// ============================================================
// CONFIGURACIÓN: cambia solo esta sección
// ============================================================

// Módulo donde está instalada ESTA cámara. Debe coincidir con el
// origen del módulo en Supabase: "modulo-1", "modulo-2", ..., "modulo-27".
const char* ORIGEN = "modulo-1";

// Backend en Render (sin https:// y sin / al final)
const char* API_HOST = "proyecto-tutoria-wu98.onrender.com";
const char* API_PATH = "/predict";

// Cada cuánto se toma y envía una foto durante el horario (milisegundos).
// 60 s es suficiente para la ocupación de un módulo y no satura el servidor
// gratuito, ni siquiera con varias cámaras enviando a la vez.
const unsigned long INTERVALO_MS = 30000;   // 30 segundos (versión de prueba)
const bool PRUEBA_SIN_HORARIO = true;       // true: envía fotos a cualquier hora

// ---------- Horario de atención (hora de Colombia) ----------
// Fuera de este horario la placa duerme y no envía fotos.
// Cada día: { hora de apertura, minuto, hora de cierre, minuto } en formato 24 h.
// Un día sin atención se deja en { 0, 0, 0, 0 }.
const int HORARIO[7][4] = {
  {  0, 0,  0, 0 },   // domingo:  cerrado
  {  6, 0, 20, 0 },   // lunes:    6:00 a. m. a 8:00 p. m.
  {  6, 0, 20, 0 },   // martes
  {  6, 0, 20, 0 },   // miércoles
  {  6, 0, 20, 0 },   // jueves
  {  6, 0, 20, 0 },   // viernes
  {  7, 0, 12, 0 },   // sábado:   7:00 a. m. a 12:00 m.
};

// Minutos antes de la apertura en que la placa despierta para "despertar" también
// al servidor de Render (que se duerme tras 15 min sin uso y tarda ~1 min en arrancar).
// Así la primera foto del día se analiza sin demora.
const int MINUTOS_ANTES_APERTURA = 3;

// Tiempo máximo esperando la respuesta (Render gratis puede tardar en despertar)
const unsigned long TIMEOUT_RESPUESTA_MS = 120000;  // 2 minutos

// Flash LED (GPIO 4). Ponlo en true si el aula está oscura (personas no detectadas).
const bool USAR_FLASH = false;

// Si fallan tantos envíos seguidos, la placa se reinicia sola.
const int MAX_FALLOS_SEGUIDOS = 5;

// ---------- Calidad de la imagen (lo que ve YOLO) ----------
// Usa SIEMPRE tamaños 4:3 para conservar todo el campo de visión del sensor:
//   FRAMESIZE_SVGA = 800x600   (más liviano)
//   FRAMESIZE_XGA  = 1024x768  (recomendado)
//   FRAMESIZE_UXGA = 1600x1200 (máximo detalle, envío más lento)
const framesize_t RESOLUCION = FRAMESIZE_XGA;

// Calidad JPEG: número MENOR = mejor calidad (10 = muy buena, 12 = buena)
const int CALIDAD_JPEG = 10;

// Fotos que se descartan antes de la buena, para que la exposición
// y el balance de blancos ya estén ajustados a la escena.
const int FOTOS_DESCARTADAS = 3;
// ============================================================

// Zona horaria de Colombia (UTC-5, sin horario de verano)
const char* ZONA_HORARIA = "<-05>5";

// Pines de la cámara AI Thinker
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#define FLASH_GPIO_NUM     4

unsigned long ultimoEnvio = 0;
int fallosSeguidos = 0;
bool primerEnvio = true;
bool horaSincronizada = false;
bool servidorDespertado = false;
unsigned long ultimoIntentoHora = 0;

// Aplica un ajuste al sensor solo si ese sensor lo soporta (evita errores).
#define AJUSTE(funcion, valor) do { if (s->funcion) s->funcion(s, valor); } while (0)

// ------------------------------------------------------------
// Cámara
// ------------------------------------------------------------
void ajustarSensor(sensor_t* s) {
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);              // igual que CameraWebServer para OV3660
  }

  // Balance de blancos automático: colores fieles.
  AJUSTE(set_whitebal, 1);
  AJUSTE(set_awb_gain, 1);
  AJUSTE(set_wb_mode, 0);            // 0 = automático

  // Exposición y ganancia automáticas, un poco hacia lo claro para que
  // las personas lejanas no queden oscuras.
  AJUSTE(set_exposure_ctrl, 1);
  AJUSTE(set_aec2, 1);               // algoritmo de exposición mejorado
  AJUSTE(set_ae_level, 1);           // -2..2 ; +1 = ligeramente más claro
  AJUSTE(set_gain_ctrl, 1);
  AJUSTE(set_gainceiling, GAINCEILING_4X);

  AJUSTE(set_brightness, 0);
  AJUSTE(set_contrast, 0);
  AJUSTE(set_saturation, 0);

  // Corrige la sombra en las esquinas del lente: así las personas de los
  // bordes se ven igual de claras que las del centro.
  AJUSTE(set_lenc, 1);

  // Corrección de píxeles defectuosos y curva de color.
  AJUSTE(set_bpc, 1);
  AJUSTE(set_wpc, 1);
  AJUSTE(set_raw_gma, 1);
}

bool iniciarCamara() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk  = XCLK_GPIO_NUM;
  config.pin_pclk  = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href  = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn  = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.jpeg_quality = CALIDAD_JPEG;

  if (psramFound()) {
    config.frame_size  = RESOLUCION;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.fb_count    = 2;
    config.grab_mode   = CAMERA_GRAB_LATEST;   // siempre la foto más reciente
  } else {
    // Sin PSRAM no hay memoria para fotos grandes: se usa VGA.
    config.frame_size  = FRAMESIZE_VGA;
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.fb_count    = 1;
    config.grab_mode   = CAMERA_GRAB_WHEN_EMPTY;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[ERROR] La cámara no inició (0x%x)\n", err);
    return false;
  }

  ajustarSensor(esp_camera_sensor_get());

  Serial.printf("[OK] Cámara lista. PSRAM: %s | Resolución: %s\n",
                psramFound() ? "sí" : "no",
                psramFound() ? "alta (XGA/configurada)" : "VGA (sin PSRAM)");
  return true;
}

camera_fb_t* tomarFoto() {
  if (USAR_FLASH) {
    digitalWrite(FLASH_GPIO_NUM, HIGH);
    delay(300);                       // tiempo para que la exposición se adapte al flash
  }

  for (int i = 0; i < FOTOS_DESCARTADAS; i++) {
    camera_fb_t* viejo = esp_camera_fb_get();
    if (viejo) esp_camera_fb_return(viejo);
    delay(100);
  }
  camera_fb_t* fb = esp_camera_fb_get();

  if (USAR_FLASH) digitalWrite(FLASH_GPIO_NUM, LOW);
  return fb;
}

// ------------------------------------------------------------
// Wi-Fi
// ------------------------------------------------------------
bool conectarWiFi() {
  if (WiFi.status() == WL_CONNECTED) return true;

  Serial.printf("Conectando a Wi-Fi \"%s\"", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long inicio = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - inicio < 20000) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[ERROR] No se pudo conectar al Wi-Fi");
    return false;
  }

  Serial.print("[OK] Wi-Fi conectado. IP: ");
  Serial.print(WiFi.localIP());
  Serial.printf("  Señal: %d dBm\n", WiFi.RSSI());
  return true;
}

// ------------------------------------------------------------
// Hora y horario de atención
// ------------------------------------------------------------
// Pide la hora a internet (NTP). Sin hora, la placa no sabe si está en
// horario, así que en ese caso sigue enviando fotos para no perder datos.
bool sincronizarHora() {
  if (horaSincronizada) return true;
  if (!primerEnvio && millis() - ultimoIntentoHora < 300000) return false;  // reintenta cada 5 min
  ultimoIntentoHora = millis();
  if (WiFi.status() != WL_CONNECTED) return false;

  Serial.println("Sincronizando la hora por internet...");
  configTzTime(ZONA_HORARIA, "pool.ntp.org", "time.google.com", "time.cloudflare.com");
  struct tm t;
  if (!getLocalTime(&t, 10000)) {
    Serial.println("[AVISO] No se obtuvo la hora. Se enviarán fotos sin revisar el horario.");
    return false;
  }
  horaSincronizada = true;
  char texto[40];
  strftime(texto, sizeof(texto), "%d/%m/%Y %H:%M:%S", &t);
  Serial.printf("[OK] Hora de Colombia: %s\n", texto);
  return true;
}

int minutosDelDia(int hora, int minuto) { return hora * 60 + minuto; }
int aperturaDe(int dia) { return minutosDelDia(HORARIO[dia][0], HORARIO[dia][1]); }
int cierreDe(int dia)   { return minutosDelDia(HORARIO[dia][2], HORARIO[dia][3]); }
bool diaConAtencion(int dia) { return cierreDe(dia) > aperturaDe(dia); }

// ---------- Festivos de Colombia (Ley 51 de 1983) ----------
// Se calculan para cualquier año, así que no hay que actualizar una lista:
// unos tienen fecha fija, otros pasan al lunes siguiente y otros dependen de la Pascua.

// Número de días desde el 1 de enero de 1970 para una fecha.
long diasDesde1970(int a, int m, int d) {
  a -= m <= 2;
  long era = (a >= 0 ? a : a - 399) / 400;
  long ae = a - era * 400;
  long dAnio = (153L * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  long de = ae * 365 + ae / 4 - ae / 100 + dAnio;
  return era * 146097 + de - 719468;
}
int diaDeLaSemana(long n) { return (int)(((n % 7) + 11) % 7); }  // 0 = domingo
long alLunes(long n) { return n + (8 - diaDeLaSemana(n)) % 7; }  // si no es lunes, el siguiente

// Domingo de Pascua (algoritmo anónimo gregoriano).
long domingoDePascua(int a) {
  int A = a % 19, b = a / 100, c = a % 100, d = b / 4, e = b % 4, f = (b + 8) / 25, g = (b - f + 1) / 3;
  int h = (19 * A + b - d - g + 15) % 30, i = c / 4, k = c % 4, l = (32 + 2 * e + 2 * i - h - k) % 7;
  int m = (A + 11 * h + 22 * l) / 451, mes = (h + l - 7 * m + 114) / 31, dia = (h + l - 7 * m + 114) % 31 + 1;
  return diasDesde1970(a, mes, dia);
}

bool esFestivo(int anio, int mes, int dia) {
  static const uint8_t FIJOS[][2]    = {{1, 1}, {5, 1}, {7, 20}, {8, 7}, {12, 8}, {12, 25}};
  static const uint8_t AL_LUNES[][2] = {{1, 6}, {3, 19}, {6, 29}, {8, 15}, {10, 12}, {11, 1}, {11, 11}};
  static const int DESDE_PASCUA[]    = {-3, -2, 43, 64, 71};  // jueves y viernes santo, Ascensión, Corpus, Sagrado Corazón
  long hoy = diasDesde1970(anio, mes, dia);
  for (auto& f : FIJOS) if (mes == f[0] && dia == f[1]) return true;
  for (auto& f : AL_LUNES) if (hoy == alLunes(diasDesde1970(anio, f[0], f[1]))) return true;
  long pascua = domingoDePascua(anio);
  for (int k : DESDE_PASCUA) if (hoy == pascua + k) return true;
  return false;
}
bool esFestivo(const struct tm& t) { return esFestivo(t.tm_year + 1900, t.tm_mon + 1, t.tm_mday); }

bool enHorario(const struct tm& t) {
  if (esFestivo(t)) return false;  // festivo: no hay clases
  if (!diaConAtencion(t.tm_wday)) return false;
  int ahora = minutosDelDia(t.tm_hour, t.tm_min);
  return ahora >= aperturaDe(t.tm_wday) && ahora < cierreDe(t.tm_wday);
}

// Segundos que faltan para la próxima apertura (busca hasta 7 días adelante).
uint64_t segundosHastaApertura(const struct tm& t) {
  long ahora = (long)minutosDelDia(t.tm_hour, t.tm_min) * 60 + t.tm_sec;
  for (int d = 0; d <= 14; d++) {  // 14 días: alcanza aunque haya festivos seguidos (Semana Santa)
    int dia = (t.tm_wday + d) % 7;
    if (!diaConAtencion(dia)) continue;
    struct tm otro = t;               // fecha de dentro de d días
    otro.tm_mday += d;
    otro.tm_hour = 12; otro.tm_isdst = -1;
    mktime(&otro);                    // normaliza fin de mes y de año
    if (esFestivo(otro)) continue;
    long faltan = (long)d * 86400 + (long)aperturaDe(dia) * 60 - ahora;
    if (faltan > 0) return (uint64_t)faltan;
  }
  return 3600;  // no debería pasar; por seguridad, revisa en una hora
}

// Pide /health al servidor para que Render arranque antes de la primera foto.
void despertarServidor() {
  WiFiClientSecure cliente;
  cliente.setInsecure();
  Serial.println("Despertando el servidor antes de la apertura...");
  if (!cliente.connect(API_HOST, 443)) {
    Serial.println("[AVISO] No se pudo contactar al servidor; se intentará con la primera foto.");
    return;
  }
  cliente.print(String("GET /health HTTP/1.1\r\nHost: ") + API_HOST + "\r\nConnection: close\r\n\r\n");
  String estado = leerLinea(cliente, millis() + TIMEOUT_RESPUESTA_MS);
  cliente.stop();
  Serial.println(estado.length() ? "[OK] Servidor listo: " + estado : String("[AVISO] El servidor no respondió a tiempo."));
}

// Apaga cámara y Wi-Fi y duerme hasta poco antes de la próxima apertura.
void dormirHastaApertura(const struct tm& t) {
  uint64_t segundos = segundosHastaApertura(t);
  // Despierta unos minutos antes para despertar también al servidor.
  const uint64_t ANTES = (uint64_t)MINUTOS_ANTES_APERTURA * 60;
  if (segundos > ANTES) segundos -= ANTES;

  // El reloj interno se desvía un poco al dormir mucho tiempo. Por eso la
  // placa duerme máximo 6 horas seguidas, despierta, vuelve a pedir la hora
  // por internet y, si aún no es la apertura, se vuelve a dormir.
  const uint64_t MAXIMO = 6ULL * 3600;
  if (segundos > MAXIMO) segundos = MAXIMO;

  if (esFestivo(t)) Serial.println("Hoy es festivo: no hay clases.");
  Serial.printf("Fuera del horario de atención. Durmiendo %llu min (hasta la apertura o para revisar la hora).\n",
                segundos / 60);
  Serial.flush();

  esp_camera_deinit();
  WiFi.disconnect(true);
  WiFi.mode(WIFI_OFF);

  // El LED del flash (GPIO 4) quedaría encendido tenue durante el sueño si no se fija en bajo.
  digitalWrite(FLASH_GPIO_NUM, LOW);
  rtc_gpio_hold_en(GPIO_NUM_4);

  esp_sleep_enable_timer_wakeup(segundos * 1000000ULL);
  esp_deep_sleep_start();
}

// ------------------------------------------------------------
// Lectura de la respuesta HTTP
// ------------------------------------------------------------
String leerLinea(WiFiClientSecure& cliente, unsigned long limite) {
  String linea;
  while (millis() < limite) {
    while (cliente.available()) {
      char c = cliente.read();
      if (c == '\n') return linea;
      if (c != '\r') linea += c;
    }
    if (!cliente.connected() && !cliente.available()) break;
    delay(5);
  }
  return linea;
}

String leerCuerpo(WiFiClientSecure& cliente, bool chunked, unsigned long limite, size_t maxBytes) {
  String cuerpo;
  if (!chunked) {
    while (millis() < limite && cuerpo.length() < maxBytes) {
      while (cliente.available() && cuerpo.length() < maxBytes) cuerpo += (char)cliente.read();
      if (!cliente.connected() && !cliente.available()) break;
      delay(5);
    }
    return cuerpo;
  }

  while (millis() < limite && cuerpo.length() < maxBytes) {
    String linea = leerLinea(cliente, limite);
    linea.trim();
    if (linea.length() == 0) continue;
    long tam = strtol(linea.c_str(), nullptr, 16);
    if (tam <= 0) break;                         // último bloque
    long leidos = 0;
    while (leidos < tam && millis() < limite) {
      if (cliente.available()) {
        char c = cliente.read();
        if (cuerpo.length() < maxBytes) cuerpo += c;
        leidos++;
      } else if (!cliente.connected()) {
        break;
      } else {
        delay(5);
      }
    }
  }
  return cuerpo;
}

// Busca "clave": numero dentro del JSON (primera aparición = nivel principal).
long valorJson(const String& json, const char* clave) {
  String patron = String("\"") + clave + "\":";
  int pos = json.indexOf(patron);
  if (pos < 0) return -1;
  pos += patron.length();
  while (pos < (int)json.length() && json[pos] == ' ') pos++;
  return json.substring(pos).toInt();
}

// ------------------------------------------------------------
// Envío de la foto a /predict
// ------------------------------------------------------------
bool enviarFoto(camera_fb_t* fb) {
  WiFiClientSecure cliente;
  // Conexión cifrada sin verificar el certificado (suficiente para el prototipo).
  cliente.setInsecure();

  Serial.printf("Conectando a %s ...\n", API_HOST);
  if (!cliente.connect(API_HOST, 443)) {
    Serial.println("[ERROR] No se pudo conectar al servidor (¿la red bloquea internet?)");
    return false;
  }

  const String limite = "----ESP32CAMLimite7MA4YWxkTrZu0gW";
  String inicio =
      String("--") + limite + "\r\n" +
      "Content-Disposition: form-data; name=\"origen\"\r\n\r\n" +
      ORIGEN + "\r\n" +
      "--" + limite + "\r\n" +
      "Content-Disposition: form-data; name=\"file\"; filename=\"captura.jpg\"\r\n"
      "Content-Type: image/jpeg\r\n\r\n";
  String fin = String("\r\n--") + limite + "--\r\n";
  size_t largoTotal = inicio.length() + fb->len + fin.length();

  cliente.print(String("POST ") + API_PATH + " HTTP/1.1\r\n");
  cliente.print(String("Host: ") + API_HOST + "\r\n");
  cliente.print("User-Agent: ESP32-CAM\r\n");
  cliente.print("Accept: application/json\r\n");
  cliente.print(String("X-API-Key: ") + API_KEY + "\r\n");   // autoriza a esta cámara
  cliente.print(String("Content-Type: multipart/form-data; boundary=") + limite + "\r\n");
  cliente.print(String("Content-Length: ") + String(largoTotal) + "\r\n");
  cliente.print("Connection: close\r\n\r\n");

  cliente.print(inicio);
  const size_t BLOQUE = 2048;
  for (size_t i = 0; i < fb->len; i += BLOQUE) {
    size_t n = (fb->len - i < BLOQUE) ? (fb->len - i) : BLOQUE;
    if (cliente.write(fb->buf + i, n) != n) {
      Serial.println("[ERROR] Se cortó la conexión enviando la foto");
      cliente.stop();
      return false;
    }
  }
  cliente.print(fin);
  Serial.printf("Foto enviada (%u bytes). Esperando análisis de YOLO...\n", fb->len);

  unsigned long limiteTiempo = millis() + TIMEOUT_RESPUESTA_MS;
  String estado = leerLinea(cliente, limiteTiempo);
  if (estado.length() == 0) {
    Serial.println("[ERROR] El servidor no respondió a tiempo");
    cliente.stop();
    return false;
  }

  int codigo = estado.substring(9, 12).toInt();   // "HTTP/1.1 200 OK"
  bool chunked = false;
  while (true) {
    String h = leerLinea(cliente, limiteTiempo);
    if (h.length() == 0) break;
    h.toLowerCase();
    if (h.startsWith("transfer-encoding:") && h.indexOf("chunked") > 0) chunked = true;
  }

  String cuerpo = leerCuerpo(cliente, chunked, limiteTiempo, 6000);
  cliente.stop();

  if (codigo == 401) {
    Serial.println("[ERROR] Clave rechazada (HTTP 401). Revisa que API_KEY en");
    Serial.println("        secretos.h sea igual a PREDICT_API_KEY en Render.");
    return false;
  }
  if (codigo == 422) {
    Serial.printf("[ERROR] El servidor no reconoce el origen \"%s\" (HTTP 422).\n", ORIGEN);
    Serial.println("        Revisa ORIGEN: debe ser modulo-1, modulo-2, ... modulo-27.");
    return false;
  }
  if (codigo == 409) {
    Serial.printf("[ERROR] El módulo \"%s\" está inactivo (HTTP 409).\n", ORIGEN);
    Serial.println("        Actívalo en el panel de administrador -> Módulos -> \"Con cámara\".");
    return false;
  }
  if (codigo != 200) {
    Serial.printf("[ERROR] El servidor respondió HTTP %d\n", codigo);
    Serial.println(cuerpo.substring(0, 300));
    return false;
  }

  Serial.println("[OK] Resultado de YOLO:");
  Serial.printf("   Personas:  %ld\n", valorJson(cuerpo, "personas"));
  Serial.printf("   Capacidad: %ld sillas\n", valorJson(cuerpo, "sillas"));
  Serial.printf("   Ocupadas:  %ld\n", valorJson(cuerpo, "ocupadas"));
  Serial.printf("   Libres:    %ld\n", valorJson(cuerpo, "libres"));
  // El backend solo responde 200 si el registro quedó guardado en Supabase.
  Serial.println("   Guardado en Supabase: sí (la página se actualizará sola)");
  return true;
}

// ------------------------------------------------------------
void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);   // evita reinicios por bajones de voltaje
  rtc_gpio_hold_dis(GPIO_NUM_4);               // libera el pin del flash tras dormir

  Serial.begin(115200);
  delay(500);
  Serial.println();
  Serial.println("==============================================");
  Serial.println(" ESP32-CAM -> YOLO  |  VERSIÓN DE PRUEBA (30 s, sin horario)");
  Serial.printf (" Módulo: %s | Una foto cada %lu s\n", ORIGEN, INTERVALO_MS / 1000);
  Serial.println("==============================================");
  if (esp_sleep_get_wakeup_cause() == ESP_SLEEP_WAKEUP_TIMER) {
    Serial.println("Despertó del modo de suspensión.");
  }

  pinMode(FLASH_GPIO_NUM, OUTPUT);
  digitalWrite(FLASH_GPIO_NUM, LOW);

  if (!iniciarCamara()) {
    Serial.println("Reiniciando en 5 s...");
    delay(5000);
    ESP.restart();
  }

  if (conectarWiFi()) sincronizarHora();
}

void loop() {
  if (!primerEnvio && millis() - ultimoEnvio < INTERVALO_MS) {
    delay(50);
    return;
  }

  // ¿Estamos dentro del horario de atención?
  if (conectarWiFi()) sincronizarHora();
  struct tm ahora;
  // Versión de prueba: no se revisa el horario ni los festivos, nunca se duerme.
  if (PRUEBA_SIN_HORARIO == false && horaSincronizada && getLocalTime(&ahora, 1000) && !enHorario(ahora)) {
    // Si faltan pocos minutos para abrir, despierta al servidor y espera despierta.
    if (segundosHastaApertura(ahora) <= (uint64_t)(MINUTOS_ANTES_APERTURA + 2) * 60) {
      if (!servidorDespertado && WiFi.status() == WL_CONNECTED) {
        despertarServidor();
        servidorDespertado = true;
      }
      delay(1000);
      return;
    }
    dormirHastaApertura(ahora);   // no regresa: la placa se reinicia al despertar
  }

  primerEnvio = false;
  ultimoEnvio = millis();

  Serial.println();
  Serial.println("---------- Nueva captura ----------");

  bool exito = false;
  if (WiFi.status() == WL_CONNECTED) {
    camera_fb_t* fb = tomarFoto();
    if (!fb) {
      Serial.println("[ERROR] No se pudo tomar la foto");
    } else {
      Serial.printf("Foto tomada: %ux%u, %u bytes\n", fb->width, fb->height, fb->len);
      exito = enviarFoto(fb);
      esp_camera_fb_return(fb);
    }
  }

  if (exito) {
    fallosSeguidos = 0;
  } else {
    fallosSeguidos++;
    Serial.printf("Fallos seguidos: %d de %d\n", fallosSeguidos, MAX_FALLOS_SEGUIDOS);
    if (fallosSeguidos >= MAX_FALLOS_SEGUIDOS) {
      Serial.println("Demasiados fallos. Reiniciando la placa...");
      delay(1000);
      ESP.restart();
    }
  }

  Serial.printf("Próxima captura en %lu s\n", INTERVALO_MS / 1000);
}
