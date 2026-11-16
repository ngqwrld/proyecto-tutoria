// Festivos de Colombia (Ley 51 de 1983, "Ley Emiliani").
// Se calculan para cualquier año: unos tienen fecha fija, otros se trasladan al
// lunes siguiente y otros dependen de la fecha de la Pascua.
(function (global) {
  const DIA = 86400000;
  const FIJOS = [[1, 1, "Año Nuevo"], [5, 1, "Día del Trabajo"], [7, 20, "Día de la Independencia"],
    [8, 7, "Batalla de Boyacá"], [12, 8, "Inmaculada Concepción"], [12, 25, "Navidad"]];
  const AL_LUNES = [[1, 6, "Día de los Reyes Magos"], [3, 19, "Día de San José"], [6, 29, "San Pedro y San Pablo"],
    [8, 15, "Asunción de la Virgen"], [10, 12, "Día de la Diversidad Étnica y Cultural"],
    [11, 1, "Día de Todos los Santos"], [11, 11, "Independencia de Cartagena"]];
  const DESDE_PASCUA = [[-3, "Jueves Santo"], [-2, "Viernes Santo"], [43, "Ascensión del Señor"],
    [64, "Corpus Christi"], [71, "Sagrado Corazón"]];

  // Domingo de Pascua (algoritmo anónimo gregoriano), en milisegundos UTC.
  function pascua(y) {
    const a = y % 19, b = Math.floor(y / 100), c = y % 100, d = Math.floor(b / 4), e = b % 4;
    const f = Math.floor((b + 8) / 25), g = Math.floor((b - f + 1) / 3), h = (19 * a + b - d - g + 15) % 30;
    const i = Math.floor(c / 4), k = c % 4, l = (32 + 2 * e + 2 * i - h - k) % 7, m = Math.floor((a + 11 * h + 22 * l) / 451);
    const mes = Math.floor((h + l - 7 * m + 114) / 31), dia = (h + l - 7 * m + 114) % 31 + 1;
    return Date.UTC(y, mes - 1, dia);
  }
  const iso = ms => new Date(ms).toISOString().slice(0, 10);
  const cache = {};

  // Mapa "AAAA-MM-DD" -> nombre del festivo.
  function festivosDelAnio(y) {
    if (cache[y]) return cache[y];
    const r = {};
    FIJOS.forEach(([m, d, n]) => { r[iso(Date.UTC(y, m - 1, d))] = n; });
    AL_LUNES.forEach(([m, d, n]) => {
      const t = Date.UTC(y, m - 1, d), w = new Date(t).getUTCDay();
      r[iso(t + ((8 - w) % 7) * DIA)] = n;            // si no cae lunes, pasa al lunes siguiente
    });
    const p = pascua(y);
    DESDE_PASCUA.forEach(([k, n]) => { r[iso(p + k * DIA)] = n; });
    return (cache[y] = r);
  }
  // Nombre del festivo de una fecha "AAAA-MM-DD", o null si es día hábil.
  function festivo(fecha) { return festivosDelAnio(Number(fecha.slice(0, 4)))[fecha] || null; }
  // Fecha de hoy en Colombia, "AAAA-MM-DD", desplazada n días.
  function fechaBogota(n = 0) {
    return new Intl.DateTimeFormat("en-CA", { timeZone: "America/Bogota" }).format(new Date(Date.now() + n * DIA));
  }

  global.Festivos = { festivo, festivosDelAnio, fechaBogota };
})(typeof window !== "undefined" ? window : globalThis);
