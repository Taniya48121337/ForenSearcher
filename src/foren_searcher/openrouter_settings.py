"""OpenRouter 的本机设置与 Windows 用户密钥加密。"""

from __future__ import annotations

import argparse
import base64
import ctypes
import json
import math
import os
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "qwen/qwen3.8-27b"
DEFAULT_CONFIG = Path(".local/openrouter.json")


def _protect(data: bytes, *, decrypt: bool = False) -> bytes:
    if os.name != "nt":
        raise RuntimeError("Encrypted key files require Windows; use OPENROUTER_API_KEY on other systems")
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]

    buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    source = Blob(len(data), buffer)
    target = Blob()
    library = ctypes.WinDLL("crypt32", use_last_error=True)
    function = library.CryptUnprotectData if decrypt else library.CryptProtectData
    function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    function.restype = wintypes.BOOL
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)):
        raise RuntimeError("Windows could not access the encrypted OpenRouter key")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        ctypes.memset(target.data, 0, target.size)
        kernel.LocalFree(target.data)


def read_settings(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    file = Path(path)
    if not file.exists():
        return {"model": DEFAULT_MODEL, "budget_usd": 1.0, "providers": ["parasail", "akashml"],
                "quantizations": ["fp8"], "reasoning_effort": "medium", "structured_outputs": False,
                "input_price_cap_per_million": 0.25, "output_price_cap_per_million": 2.55}
    try:
        value = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError
        budget = float(value.get("budget_usd", 1.0))
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError
        return {**value, "budget_usd": budget}
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError("OpenRouter settings could not be read; reopen the settings window") from exc


def key_configured(path: str | Path = DEFAULT_CONFIG) -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY", "").strip() or read_settings(path).get("encrypted_key"))


def load_api_key(path: str | Path = DEFAULT_CONFIG) -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    value = read_settings(path).get("encrypted_key")
    if not isinstance(value, str) or not value:
        raise RuntimeError("Configure OpenRouter with python -m foren_searcher.openrouter_settings")
    try:
        return _protect(base64.b64decode(value, validate=True), decrypt=True).decode("utf-8")
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError("The encrypted OpenRouter key could not be read") from exc


def save_settings(path: str | Path, *, key: str, model: str, budget_usd: float,
                  visual_model: str | None = None) -> None:
    if not model.strip() or not math.isfinite(budget_usd) or budget_usd <= 0:
        raise ValueError("Enter a model ID and a positive total budget")
    value = read_settings(path)
    if key.strip():
        value["encrypted_key"] = base64.b64encode(_protect(key.strip().encode("utf-8"))).decode("ascii")
    elif not value.get("encrypted_key") and not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("Enter an OpenRouter API key")
    value.update(model=model.strip(), budget_usd=budget_usd)
    if visual_model is not None:
        if not visual_model.strip():
            raise ValueError("Enter a visual model ID")
        value["visual_model"] = visual_model.strip()
    else:
        value.pop("visual_model", None)
        value.pop("visual_settings", None)
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(file)


def main() -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    parser = argparse.ArgumentParser(description="Configure OpenRouter on this computer")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    settings = read_settings(args.config)
    window = tk.Tk()
    window.title("ForenSearcher · OpenRouter 设置")
    window.resizable(False, False)
    frame = ttk.Frame(window, padding=24)
    frame.grid()
    ttk.Label(frame, text="OpenRouter API Key", font=("Microsoft YaHei UI", 13)).grid(row=0, column=0, sticky="w", pady=(0, 8))
    key_input = ttk.Entry(frame, show="●", width=58)
    key_input.grid(row=1, column=0, sticky="ew")
    hint = "已有密钥，留空可保留。" if key_configured(args.config) else "在此粘贴密钥。密钥以 Windows 当前用户加密保存。"
    ttk.Label(frame, text=hint).grid(row=2, column=0, sticky="w", pady=(6, 18))
    ttk.Label(frame, text="调查模型 ID（GIA / HI）").grid(row=3, column=0, sticky="w")
    model_input = ttk.Entry(frame, width=58)
    model_input.insert(0, settings.get("model", DEFAULT_MODEL))
    model_input.grid(row=4, column=0, sticky="ew", pady=(6, 18))
    ttk.Label(frame, text="GIA 与各 Workspace 的 HI 共用此模型；HI 直接接收图片。", justify="left").grid(row=5, column=0, sticky="w", pady=(0, 18))
    ttk.Label(frame, text="本项目累计推理预算（美元）").grid(row=7, column=0, sticky="w")
    budget_input = ttk.Entry(frame, width=18)
    budget_input.insert(0, str(settings["budget_usd"]))
    budget_input.grid(row=8, column=0, sticky="w", pady=(6, 8))
    ttk.Label(frame, text="所有角色共用累计预算；保存设置不会清空已记录费用。\n建议同时在 OpenRouter 为此 Key 设置额度上限。\n保存设置本身不发送推理请求。", justify="left").grid(row=9, column=0, sticky="w", pady=(0, 16))

    def save() -> None:
        try:
            save_settings(args.config, key=key_input.get(), model=model_input.get(),
                          budget_usd=float(budget_input.get()))
        except Exception:
            messagebox.showerror("未保存", "请检查密钥、模型 ID 和预算；预算必须为正数。", parent=window)
            return
        key_input.delete(0, tk.END)
        messagebox.showinfo("已保存", "OpenRouter 设置已保存在本机。", parent=window)
        window.destroy()

    ttk.Button(frame, text="保存设置", command=save).grid(row=10, column=0, sticky="e")
    key_input.focus_set()
    window.mainloop()


if __name__ == "__main__":
    main()
