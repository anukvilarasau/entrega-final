# Pruebas finales de punta a punta (API real en Docker)

| # | Escenario | Estado | Turno | Pausa HITL | Permiso web | Fuentes guía / memoria / web | Búsquedas web | Entra en el presupuesto | Validación | Tiempo de grafo (s) | Tokens | Costo (US$) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [solo_guias](1_solo_guias.json) | DONE | 1 | no | sin_pedir | 6 / 0 / 0 | 0 | — | no se validó | 23.27 | 25846 | 0.004063 |
| 2 | [flujo_completo](2_flujo_completo.json) | DONE | 1 | sí | aprobada | 17 / 0 / 12 | 4 | sí | aprobada | 74.02 | 104767 | 0.01628 |
| 3 | [seguimiento_tras_reinicio](3_seguimiento_tras_reinicio.json) | DONE | 2 | sí | aprobada | 7 / 9 / 3 | 1 | sí | aprobada | 62.48 | 102004 | 0.015207 |
| 4 | [no_alcanza](4_no_alcanza.json) | DONE | 1 | no | sin_pedir | 10 / 13 / 0 | 0 | **no** | aprobada | 64.96 | 108940 | 0.016108 |
| 5 | [error_controlado](5_error_controlado.json) | DONE | 1 | sí | rechazada | 12 / 3 / 0 | 0 | — | rechazada | 67.14 | 108354 | 0.015437 |

Fuentes: cantidad de fuentes de cada origen que citó el investigador. Tiempo de grafo: solo ejecución (sin la espera humana). Costo: tokens × precios de observability/monitoreo.py.

- **1. solo_guias** — RAG sin pausa: lo estable sale de las guías (0 búsquedas web, sin analista). LangSmith: run `01a213fe-a245-747c-b948-155c39f627d4`, 1 tramo(s), latencia 23.25 s, costo US$ 0.004063.
- **2. flujo_completo** — Flujo completo: guías + pausa HITL + web + analista + síntesis. LangSmith: run `01a213ff-02b9-7551-ad3f-3dd5faf9bb34`, 2 tramo(s), latencia 74.02 s, costo US$ 0.01628.
- **3. seguimiento_tras_reinicio** — Follow-up después de `docker compose restart app`: recuerda el pedido (checkpoint en Redis). LangSmith: run `01a21400-69fc-77b2-a10d-b424d9424679`, 2 tramo(s), latencia 62.46 s, costo US$ 0.015207. Reinicio de la app: 16.0 s.
- **4. no_alcanza** — Presupuesto insuficiente: el analista lo detecta y propone una alternativa. LangSmith: run `01a21405-e663-74da-98b1-f76793d4483b`, 1 tramo(s), latencia 64.96 s, costo US$ 0.016108.
- **5. error_controlado** — Error controlado: fechas imposibles + web rechazada → DONE pidiendo corregir. LangSmith: run `01a21402-efeb-73c8-b8f0-4ff1c37d392a`, 2 tramo(s), latencia 67.13 s, costo US$ 0.015437. Errores HTTP: {'GET job inexistente (404)': 404, 'seguimiento de thread inexistente (404)': 404, 'pregunta demasiado corta (422)': 422, 'campo desconocido (422)': 422}.
