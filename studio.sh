#!/usr/bin/env bash
# Emma Studio — Stack-Steuerung
#
# Startet/stoppt den kompletten Studio-Stack in Abhängigkeitsreihenfolge:
#   1. Gemma-4-LLM   :8080  → ~/models/gemma4_26b/launch-llama-server.sh   (GPU0)
#   2. OmniVoice TTS :8880  → systemd --user omnivoice.service             (GPU1)
#   3. Emma Studio   :8890  → systemd --user emma-studio.service
#
# Kein Autostart: die Units sind bewusst *disabled* — der Stack belegt VRAM nur
# auf Zuruf. Geteilte Dienste (SearXNG :8081, bge-m3 :7777, Dashboard) werden
# NICHT angefasst, die gehören nicht zum Studio; `status` zeigt nur, ob sie
# erreichbar sind (Emma braucht sie für Fakten- und Web-Suche).
#
# Usage: ./studio.sh {start|stop|restart|status} [--no-llm]
#   --no-llm   nur TTS + Studio starten (Cloud-LLM, oder wenn der VRAM für ein
#              anderes Modell gebraucht wird)

set -u

# Das LLM wird über sein eigenes Skript gesteuert, nicht über eine Unit: es kennt
# GPU-Wahl, KV-Cache-Typ und Kontext-Autosizing. (llama-llm.service ist veraltet:
# zeigt auf verschobene Modellpfade und würde GPU1 statt GPU0 belegen.)
LLM_CTL="${LLM_CTL:-$HOME/models/gemma4_26b/launch-llama-server.sh}"
LLM_HEALTH="http://localhost:8080/health"
LLM_LOG="$HOME/models/gemma4_26b/llama-server.log"

SERVICES=(omnivoice emma-studio)   # Start in dieser Reihenfolge, Stopp rückwärts

c_ok()  { printf '\033[32m%s\033[0m\n' "$*"; }
c_err() { printf '\033[31m%s\033[0m\n' "$*"; }
c_dim() { printf '\033[2m%s\033[0m\n' "$*"; }

reaches() { curl -sk -o /dev/null --max-time 2 "$1"; }

llm_running() { [ -x "$LLM_CTL" ] && "$LLM_CTL" status >/dev/null 2>&1; }

start_llm() {
    if llm_running; then c_dim "Gemma-LLM läuft bereits (:8080)"; return 0; fi
    if [ ! -x "$LLM_CTL" ]; then c_err "LLM-Steuerskript nicht ausführbar: $LLM_CTL"; return 1; fi
    echo "Starte Gemma-LLM (:8080, GPU0, Kontext nach freiem VRAM) …"
    # Das Skript wartet selbst auf "server is listening" (READY_TIMEOUT 600 s).
    "$LLM_CTL" start || { c_err "LLM-Start fehlgeschlagen — Log: $LLM_LOG"; return 1; }
}

stop_llm() {
    if ! llm_running; then c_dim "Gemma-LLM ist aus"; return 0; fi
    echo "Stoppe Gemma-LLM (:8080) …"
    "$LLM_CTL" stop || c_err "LLM-Stopp meldete einen Fehler — $LLM_CTL status prüfen"
}

wait_healthy() {
    # wartet bis die Ports antworten (max ~90 s, OmniVoice lädt sein CUDA-Modell)
    local want_llm="${1:-1}" deadline=$((SECONDS + 90))
    while (( SECONDS < deadline )); do
        if reaches https://localhost:8890/ && reaches http://localhost:8880/ \
           && { (( ! want_llm )) || reaches "$LLM_HEALTH"; }; then
            return 0
        fi
        sleep 2
    done
    return 1
}

do_start() {
    local want_llm=1 was_down=0 s

    [ "${1:-}" = "--no-llm" ] && want_llm=0
    if (( want_llm )); then
        start_llm || c_err "  → Stack startet ohne Sprachmodell (Emma kann nicht antworten) — nachziehen mit: $LLM_CTL start"
    else
        c_dim "LLM übersprungen (--no-llm)"
    fi

    for s in "${SERVICES[@]}"; do
        if systemctl --user is-active --quiet "$s"; then
            c_dim "$s läuft bereits"
        else
            echo "Starte $s …"
            systemctl --user start "$s" || { c_err "$s: Start fehlgeschlagen"; exit 1; }
            was_down=1
        fi
    done

    (( was_down )) && {
        echo "Warte auf Health-Checks (:8890 https, :8880 tts$((( want_llm )) && echo ', :8080 llm')) …"
        wait_healthy "$want_llm" \
            && c_ok "✓ Studio-Stack online" \
            || c_err "⚠ Services laufen, aber Health-Check timeout — journalctl --user -u emma-studio/-u omnivoice ansehen"
    }
}

do_stop() {
    # rückwärts: erst Studio (nimmt keine Requests mehr an), dann TTS, dann LLM
    local i s
    for (( i=${#SERVICES[@]}-1 ; i>=0 ; i-- )); do
        s=${SERVICES[$i]}
        systemctl --user is-active --quiet "$s" || { c_dim "$s ist aus"; continue; }
        echo "Stoppe $s …"
        systemctl --user stop "$s"
    done
    stop_llm
    c_ok "✓ Studio-Stack gestoppt"
}

do_status() {
    local s llm
    if llm_running; then llm="$("$LLM_CTL" status 2>&1 | head -1)"; else llm="aus"; fi
    printf '%-14s %s\n' "gemma-llm:" "$llm"
    for s in "${SERVICES[@]}"; do
        printf '%-14s %s (%s)\n' "$s:" "$(systemctl --user is-active "$s")" "$(systemctl --user is-enabled "$s" 2>/dev/null)"
    done
    echo
    reaches https://localhost:8890/ && c_ok '  :8890 Studio antwortet' || c_err '  :8890 keine Antwort'
    reaches http://localhost:8880/ && c_ok '  :8880 TTS antwortet'    || c_err '  :8880 keine Antwort'
    reaches "$LLM_HEALTH"          && c_ok '  :8080 LLM antwortet'    || c_err '  :8080 keine Antwort'
    echo
    c_dim "  geteilte Dienste (gehören nicht zum Studio, werden nie angefasst):"
    reaches http://localhost:7777/health && c_dim '  :7777 bge-m3  erreichbar' \
                                         || c_err '  :7777 bge-m3  fehlt — Fakten-Suche degradiert (systemctl --user start llama-embed)'
    reaches http://localhost:8081/ && c_dim '  :8081 SearXNG erreichbar' \
                                   || c_err '  :8081 SearXNG fehlt — Web-Fallback fehlt'
}

cmd="${1:-}"
shift || true
case "$cmd" in
    start)   do_start "${1:-}" ;;
    stop)    do_stop ;;
    restart) do_stop; sleep 1; do_start "${1:-}" ;;
    status)  do_status ;;
    *) echo "Usage: $0 {start|stop|restart|status} [--no-llm]"; exit 2 ;;
esac
