"""
Abre la app de Streamlit Community Cloud con un navegador real (Playwright)
para que cuente como visita. Si la app esta dormida, hace click en el boton
de despertar y espera a que vuelva a cargar.

Uso: python scripts/keep_awake.py https://tu-app.streamlit.app
"""

import re
import sys

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

WAKE_BUTTON = re.compile(r"get this app back up", re.IGNORECASE)


def main(url: str) -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(10_000)

        button = page.get_by_role("button", name=WAKE_BUTTON)
        if button.count() > 0:
            print(f"DORMIDA -> despertando {url}")
            button.first.click()
            try:
                # El boton desaparece cuando la app arranca
                button.first.wait_for(state="detached", timeout=180_000)
            except PWTimeout:
                print("La app no termino de despertar en 3 minutos")
                browser.close()
                return 1
            page.wait_for_timeout(20_000)
            print("DESPIERTA")
        else:
            # Ya estaba activa: dejamos la sesion abierta un rato para que
            # el websocket conecte y la visita se registre
            page.wait_for_timeout(20_000)
            print(f"OK, ya estaba activa: {url}")

        browser.close()
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Uso: python keep_awake.py <URL de la app>")
    sys.exit(main(sys.argv[1]))
