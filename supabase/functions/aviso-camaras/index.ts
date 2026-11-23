// Revisa cada 5 minutos (pg_cron) si alguna cámara dejó de enviar lecturas en el horario
// de atención y avisa por correo a los administradores. También avisa cuando se recupera.
// La función no recibe datos de quien la llama: todo lo consulta ella misma, así que
// llamarla de más no genera correos repetidos (cada caída se avisa una sola vez).
import { createClient } from "npm:@supabase/supabase-js@2";
import nodemailer from "npm:nodemailer@6.9.14";
import { festivo, fechaBogota } from "./festivos.ts";

const sb = createClient(Deno.env.get("SUPABASE_URL")!, Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!);
const REMITENTE = "monitortutorias.uts@gmail.com";
const MIN_SIN_DATOS = 15; // minutos sin lecturas para considerar caída la cámara
// Horario de la sede (hora de Colombia), en minutos del día: [apertura, cierre]
const HORARIO: Record<number, [number, number]> = {
  1: [360, 1200], 2: [360, 1200], 3: [360, 1200], 4: [360, 1200], 5: [360, 1200], 6: [420, 720],
};

function ahoraBogota() {
  const d = new Date(Date.now() - 5 * 3600 * 1000); // Colombia: UTC-5 sin horario de verano
  return { dia: d.getUTCDay(), min: d.getUTCHours() * 60 + d.getUTCMinutes() };
}
const fecha = (iso: string | null) => iso
  ? new Date(iso).toLocaleString("es-CO", { timeZone: "America/Bogota", dateStyle: "long", timeStyle: "short" })
  : "nunca";

async function enviar(para: string[], asunto: string, cuerpo: string) {
  const clave = (Deno.env.get("SMTP_PASS") ?? "").replace(/\s+/g, "");
  if (!clave) throw new Error("Falta el secreto SMTP_PASS");
  const t = nodemailer.createTransport({ host: "smtp.gmail.com", port: 465, secure: true, auth: { user: REMITENTE, pass: clave } });
  const html = `<div style="font-family:Arial,sans-serif;font-size:15px;color:#14231a;max-width:560px">
    <div style="background:#1f6b3a;color:#fff;padding:14px 18px;border-radius:10px 10px 0 0;font-weight:700">Monitor de módulos de tutoría · UTS</div>
    <div style="border:1px solid #d8e2d8;border-top:0;padding:18px;border-radius:0 0 10px 10px;line-height:1.5">${cuerpo}</div>
    <p style="font-size:12px;color:#6b7a70">Mensaje automático del sistema. No es necesario responder.</p></div>`;
  await t.sendMail({ from: `Monitor de tutorías UTS <${REMITENTE}>`, to: para.join(", "), subject: asunto, html });
}

Deno.serve(async (req) => {
  try {
    const url = new URL(req.url);
    const { data: admins } = await sb.from("perfiles").select("correo").eq("rol", "admin").eq("activo", true);
    const para = (admins ?? []).map((a) => a.correo);
    if (!para.length) return Response.json({ ok: true, nota: "sin administradores activos" });

    // Prueba manual: ?prueba=<token guardado en config_avisos>
    const prueba = url.searchParams.get("prueba");
    if (prueba) {
      const { data: cfg } = await sb.from("config_avisos").select("token_prueba").eq("id", 1).single();
      if (!cfg || prueba !== cfg.token_prueba) return Response.json({ ok: false, error: "token inválido" }, { status: 403 });
      await enviar(para, "Prueba: avisos de cámaras activados",
        `<p>Este es un correo de prueba.</p><p>A partir de ahora recibirás un aviso cuando un módulo con cámara pase <b>${MIN_SIN_DATOS} minutos</b> sin enviar lecturas durante el horario de atención, y otro cuando vuelva a funcionar.</p>`);
      return Response.json({ ok: true, prueba: "enviada", para });
    }

    const { dia, min } = ahoraBogota();
    const h = HORARIO[dia];
    // Los festivos no hay clases y las cámaras duermen todo el día: no se revisa nada.
    const hoyFestivo = festivo(fechaBogota());
    // Se da un margen de MIN_SIN_DATOS tras la apertura para que llegue la primera lectura.
    const enHorario = !hoyFestivo && !!h && min >= h[0] + MIN_SIN_DATOS && min < h[1];

    const { data: modulos } = await sb.from("modulos").select("id, nombre").eq("activo", true);
    const { data: abiertas } = await sb.from("alertas_camara").select("id, modulo_id, creada_en").is("recuperada_en", null);
    const resumen: unknown[] = [];

    for (const m of modulos ?? []) {
      const { data: ult } = await sb.from("registros_ocupacion").select("fecha_hora").eq("modulo_id", m.id)
        .order("fecha_hora", { ascending: false }).limit(1).maybeSingle();
      const ultima = ult?.fecha_hora ?? null;
      const minSin = ultima ? (Date.now() - new Date(ultima).getTime()) / 60000 : Infinity;
      const abierta = (abiertas ?? []).find((a) => a.modulo_id === m.id);

      if (abierta && ultima && new Date(ultima) > new Date(abierta.creada_en)) {
        await sb.from("alertas_camara").update({ recuperada_en: new Date().toISOString() }).eq("id", abierta.id);
        await enviar(para, `✅ ${m.nombre}: la cámara volvió a enviar lecturas`,
          `<p>La cámara del <b>${m.nombre}</b> volvió a funcionar.</p><p>Última lectura: <b>${fecha(ultima)}</b></p>`);
        resumen.push({ modulo: m.id, accion: "recuperada" });
      } else if (!abierta && enHorario && minSin >= MIN_SIN_DATOS) {
        // Se registra primero (el índice único evita avisos dobles) y, si el correo falla,
        // se borra el registro para que la siguiente revisión lo vuelva a intentar.
        const { data: nueva, error } = await sb.from("alertas_camara").insert({ modulo_id: m.id, ultima_lectura: ultima }).select("id").single();
        if (!error && nueva) {
          try {
            await enviar(para, `⚠️ ${m.nombre}: la cámara no está enviando lecturas`,
              `<p>La cámara del <b>${m.nombre}</b> lleva más de <b>${MIN_SIN_DATOS} minutos</b> sin enviar lecturas durante el horario de atención.</p>
               <p>Última lectura recibida: <b>${fecha(ultima)}</b></p>
               <p>Revisa que la placa tenga energía y conexión a internet. Te avisaremos cuando vuelva a funcionar.</p>`);
            resumen.push({ modulo: m.id, accion: "alerta" });
          } catch (e) {
            await sb.from("alertas_camara").delete().eq("id", nueva.id);
            throw e;
          }
        }
      }
    }
    return Response.json({ ok: true, enHorario, festivo: hoyFestivo, resumen });
  } catch (e) {
    console.error(e);
    return Response.json({ ok: false, error: String(e) }, { status: 500 });
  }
});
