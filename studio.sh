#!/usr/bin/env bash
# Emma Studio — Stack-Steuerung
#
# Startet/stoppt den kompletten Studio-Stack in Abhängigkeitsreihenfolge:
#   1. Gemma-4-LLM   :8080  → local llama-server launcher (GPU0)
#   2. OmniVoice TTS :8880  → systemd --user omnivoice.service             (GPU1)
#   3. Emma Studio   :8890  → systemd --user emma-studio.service
#
# Kein Autostart: die Units sind bewusst *disabled* — der Stack belegt VRAM nur
# auf Zuruf. Geteilte Dienste (SearXNG :8081, bge-m3 :7777, Dashboard) werden
# NICHT angefasst, die gehören nicht zum Studio; `status` zeigt nur, ob sie
# erreichbar sind (Emma braucht sie für Fakten- und Web-Suche).
#
# Vor dem Start prüft das Skript, wem die Stack-Ports gehören: hält ein
# Fremdprozess einen davon (typisch: ein vergessener Dev-Server
# `python -m studio --port 8890`), kann die Unit nicht binden, läuft in eine
# systemd-Restart-Schleife und der Health-Check wartet 90 s ins Leere. In dem
# Fall bricht `start` sofort mit PID und Kommandozeile des Halters ab.
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

LLM_PORT=8080
TTS_PORT=8880
STUDIO_PORT=8890

c_ok()  { printf '\033[32m%s\033[0m\n' "$*"; }
c_err() { printf '\033[31m%s\033[0m\n' "$*"; }
c_dim() { printf '\033[2m%s\033[0m\n' "$*"; }

reaches() { curl -sk -o /dev/null --max-time 2 "$1"; }

llm_running() { [ -x "$LLM_CTL" ] && "$LLM_CTL" status >/dev/null 2>&1; }

# ── Port-Diagnose ───────────────────────────────────────────────────────────
# Der Stack bindet feste Ports. Hält ein Fremdprozess einen davon, scheitert der
# Bind, die Unit läuft in eine systemd-Restart-Schleife und der Health-Check
# wartet 90 s ins Leere. Deshalb: vor dem Start prüfen, wem die Ports gehören.

port_pids()   { ss -Hltnp "sport = :$1" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u; }
port_busy()   { ss -Hltn "sport = :$1" 2>/dev/null | grep -q .; }
pid_cmdline() { tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null; }

# Hält ein Prozess der Unit $1 den Port $2? (über cgroup-Zugehörigkeit, nicht Name)
unit_owns_port() {
    local pid
    for pid in $(port_pids "$2"); do
        case "$(cat "/proc/$pid/cgroup" 2>/dev/null)" in *"/$1") return 0 ;; esac
    done
    return 1
}

port_owners() {   # Halter eines Ports ausgeben (leer, wenn frei)
    local pid
    for pid in $(port_pids "$1"); do printf '    pid %s: %s\n' "$pid" "$(pid_cmdline "$pid")"; done
    port_busy "$1" && [ -z "$(port_pids "$1")" ] && printf '    (Halter nicht sichtbar — fremder Benutzer?)\n'
    return 0
}

# 0 = Port frei oder gehört der Unit $2 ($2 leer => nur "frei" zählt)
port_ready() {
    port_busy "$1" || return 0
    [ -n "$2" ] && unit_owns_port "$2" "$1"
}

require_port() {   # $1=Port $2=Unit $3=Label $4=Hinweis
    port_ready "$1" "$2" && return 0
    c_err "Port $1 ($3) ist von einem Fremdprozess belegt:"
    port_owners "$1"
    c_err "  → $4"
    return 1
}

# Meldet nur Konflikte — Ports, die der zuständigen Unit bzw. dem LLM fehlen.
report_conflicts() {
    local conflict=0
    if ! port_ready "$STUDIO_PORT" emma-studio.service; then
        c_err "  :$STUDIO_PORT wird nicht von emma-studio.service gehalten:"
        port_owners "$STUDIO_PORT"; conflict=1
    fi
    if ! port_ready "$TTS_PORT" omnivoice.service; then
        c_err "  :$TTS_PORT wird nicht von omnivoice.service gehalten:"
        port_owners "$TTS_PORT"; conflict=1
    fi
    if ! llm_running && ! port_ready "$LLM_PORT" ""; then
        c_err "  :$LLM_PORT ist belegt, aber $LLM_CTL meldet 'aus':"
        port_owners "$LLM_PORT"; conflict=1
    fi
    return $(( conflict ? 1 : 0 ))
}

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

restarts() { systemctl --user show -p NRestarts --value "$1" 2>/dev/null; }

healthy_now() {
    local want_llm="${1:-1}"
    reaches "https://localhost:$STUDIO_PORT/" || return 1
    reaches "http://localhost:$TTS_PORT/" || return 1
    (( want_llm )) && ! reaches "$LLM_HEALTH" && return 1
    return 0
}

wait_healthy() {
    # wartet bis die Ports antworten (max ~90 s, OmniVoice lädt sein CUDA-Modell).
    # Läuft eine Unit in die Restart-Schleife (z.B. Bind-Konflikt), bricht das
    # hier nach wenigen Sekunden ab, statt 90 s ins Leere zu warten.
    local want_llm="${1:-1}" deadline=$((SECONDS + 90)) s st n
    while (( SECONDS < deadline )); do
        healthy_now "$want_llm" && return 0
        for s in "${SERVICES[@]}"; do
            st=$(systemctl --user is-active "$s" 2>/dev/null)
            n=$(restarts "$s"); [[ $n =~ ^[0-9]+$ ]] || n=0
            if [ "$st" = failed ] || (( n > 2 )); then
                c_err "  $s kommt nicht hoch (Status: $st, Neustarts: $n)"
                return 1
            fi
        done
        if (( want_llm )) && ! llm_running; then
            c_err "  Gemma-LLM ist nicht mehr erreichbar — Log: $LLM_LOG"
            return 1
        fi
        sleep 2
    done
    return 1
}

do_start() {
    local want_llm=1 was_down=0 s conflict=0 bad=0

    [ "${1:-}" = "--no-llm" ] && want_llm=0

    # Vorflug: die Ports müssen frei bzw. in unserer Hand sein. Sonst startet der
    # Stack blind in einen Bind-Konflikt (LLM lädt 23 GB VRAM für nichts).
    if (( want_llm )) && ! llm_running; then
        require_port "$LLM_PORT" "" "Gemma-LLM" \
            "laufenden llama-server stoppen ($LLM_CTL stop) oder mit --no-llm starten." || conflict=1
    fi
    require_port "$TTS_PORT" omnivoice.service "OmniVoice-TTS" \
        "fremden Prozess beenden (z.B. altes start_omnivoice.sh)." || conflict=1
    require_port "$STUDIO_PORT" emma-studio.service "Emma-Studio" \
        "fremden Prozess beenden (typisch ein vergessener Dev-Server: python -m studio --port $STUDIO_PORT)." || conflict=1
    if (( conflict )); then
        c_err "Start abgebrochen — es wurde nichts gestartet."
        exit 1
    fi

    if (( want_llm )); then
        start_llm || { c_err "  → Stack startet ohne Sprachmodell (Emma kann nicht antworten) — nachziehen mit: $LLM_CTL start"; want_llm=0; }
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
        if wait_healthy "$want_llm"; then
            c_ok "✓ Studio-Stack online"
        else
            c_err "⚠ Health-Check fehlgeschlagen:"
            for s in "${SERVICES[@]}"; do
                systemctl --user is-active --quiet "$s" && continue
                bad=1
                c_err "  $s ($(systemctl --user is-active "$s")) — letzte Zeilen:"
                journalctl --user -u "$s" -n 12 --no-pager 2>/dev/null | sed 's/^/    /'
            done
            (( bad )) || c_err "  Units sind aktiv, antworten aber nicht — journalctl --user -u emma-studio -u omnivoice prüfen"
            (( want_llm )) && c_dim "  LLM-Log: $LLM_LOG"
        fi
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
    reaches "https://localhost:$STUDIO_PORT/" && c_ok "  :$STUDIO_PORT Studio antwortet" || c_err "  :$STUDIO_PORT keine Antwort"
    reaches "http://localhost:$TTS_PORT/"    && c_ok "  :$TTS_PORT TTS antwortet"       || c_err "  :$TTS_PORT keine Antwort"
    reaches "$LLM_HEALTH"                    && c_ok "  :$LLM_PORT LLM antwortet"       || c_err "  :$LLM_PORT keine Antwort"
    echo
    # Fremdprozess auf einem Stack-Port = Bind-Konflikt beim nächsten `start`.
    report_conflicts || echo
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
