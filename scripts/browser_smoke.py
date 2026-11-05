"""Optional UI smoke test. Needs Playwright and a running local app."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--browser", default="/usr/bin/chromium")
    args = parser.parse_args()
    output = Path(".local")
    output.mkdir(exist_ok=True)
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.browser, args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": 1440, "height": 1080}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until="networkidle")
        page.locator(".document-item").first.wait_for()
        assert page.locator(".document-item input:checked").count() == 1
        page.screenshot(path=str(output / "workspace-desktop.png"), full_page=True)
        page.get_by_role("button", name="经营质量分析 ↗").click()
        page.get_by_role("button", name="开始研究").click()
        page.locator("#results").wait_for(state="visible")
        assert page.locator(".calculation").count() == 5
        assert page.locator(".trace-step").count() == 6
        assert page.locator(".source-card").count() == 3
        assert "14.40" in page.locator("#calculations").inner_text()
        page.locator(".calculation summary").first.click()
        assert "125000" in page.locator(".calculation").first.inner_text()
        assert "单位依据" in page.locator(".calculation").first.inner_text()
        page.locator(".citation").first.click()
        assert page.locator(".source-card.highlight").count() == 1
        with page.expect_download() as download:
            page.get_by_role("button", name="导出 Markdown ↓").click()
        assert download.value.suggested_filename.endswith(".md")
        exported = Path(download.value.path()).read_text(encoding="utf-8")
        assert "离线演示（非模型生成）" in exported and "25.00" in exported
        page.screenshot(path=str(output / "research-desktop.png"), full_page=True)
        page.get_by_role("button", name="项目架构").click()
        assert page.locator("#architecture-view").is_visible()
        page.get_by_role("button", name="研究工作台").click()
        page.get_by_role("button", name="检验证据边界 ↗").click()
        page.get_by_role("button", name="开始研究").click()
        page.locator("#results").wait_for(state="visible")
        assert "证据不足" in page.locator("#answer").inner_text()
        assert page.locator(".calculation").count() == 0
        page.set_input_files("#upload", {"name": "ui-smoke.txt", "mimeType": "text/plain", "buffer": "营业收入为 300 万元。回款风险需要关注。".encode()})
        page.locator(".document-item", has_text="ui-smoke.txt").wait_for()
        assert page.locator(".document-item input:checked").count() == 1
        page.locator("#question").fill("营业收入是多少")
        page.get_by_role("button", name="开始研究").click()
        page.locator("#results").wait_for(state="visible")
        assert "300" in page.locator("#answer").inner_text()
        assert "125000" not in page.locator("#answer").inner_text()
        assert page.locator(".source-card").count() == 1
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator("#toast").wait_for(state="hidden")
        page.screenshot(path=str(output / "workspace-mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "mobile horizontal overflow"
        # Exercise partial-result presentation with a deterministic response fixture.
        partial = page.request.post(args.url + "/api/research", data={"question": "分析经营质量", "document_ids": ["demo-qiheng-2024"], "mode": "demo"}).json()
        partial.update(mode="live", status="partial", answer_origin="evidence_summary",
                       answer="研究未完整完成，以下为工具结果摘录：收入同比 25.00%。 [demo-qiheng-2024:p1:c1]",
                       warnings=["检测到重复工具请求，已停止循环。本次为部分结果。"])
        for checkbox in page.locator(".document-item input").all():
            checkbox.set_checked(False)
        page.locator(".document-item input").first.set_checked(True)
        page.route("**/api/research", lambda route: route.fulfill(json=partial))
        page.get_by_role("button", name="开始研究").click()
        page.locator("#results").wait_for(state="visible")
        assert "部分结果" in page.locator("#result-meta").inner_text()
        assert "工具结果摘录" in page.locator("#answer-mode").inner_text()
        assert "已停止循环" in page.locator("#warnings").inner_text()
        with page.expect_download() as partial_download:
            page.get_by_role("button", name="导出 Markdown ↓").click()
        assert "完成状态：部分结果" in Path(partial_download.value.path()).read_text(encoding="utf-8")
        page.screenshot(path=str(output / "partial-result-mobile.png"), full_page=True)
        assert not errors, errors
        browser.close()
    print(json.dumps({"status": "passed", "checks": ["desktop", "demo calculations", "trace", "source jump", "export", "architecture", "evidence refusal", "upload and isolation", "mobile layout", "partial-result fixture and export", "no JavaScript errors"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
