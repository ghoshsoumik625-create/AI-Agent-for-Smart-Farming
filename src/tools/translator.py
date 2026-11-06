"""
translator.py — Language detection and translation using IBM Watson Language Translator.
Falls back to deep-translator (Google Translate) when IBM credentials are absent.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Language name → ISO code mapping for convenience
LANGUAGE_MAP: dict[str, str] = {
    "english": "en",
    "hindi": "hi",
    "telugu": "te",
    "tamil": "ta",
    "kannada": "kn",
    "marathi": "mr",
    "bengali": "bn",
    "punjabi": "pa",
    "gujarati": "gu",
    "malayalam": "ml",
    "odia": "or",
    "urdu": "ur",
    "assamese": "as",
}

# ISO code → readable name
CODE_TO_NAME: dict[str, str] = {v: k.capitalize() for k, v in LANGUAGE_MAP.items()}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """Return ISO 639-1 language code detected from input text."""
    try:
        from langdetect import detect

        code = detect(text)
        logger.debug("Detected language: %s for text: %.50s", code, text)
        return code
    except Exception as e:
        logger.warning("Language detection failed: %s — defaulting to 'en'", e)
        return "en"


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate_text(text: str, target_lang: str = "en", source_lang: str = "auto") -> str:
    """
    Translate text to target_lang.
    Uses IBM Watson Translator when credentials are available,
    otherwise falls back to deep-translator (Google Translate).
    """
    if not text.strip():
        return text

    # Try IBM Watson Translator first
    api_key = settings.ibm_translator_api_key
    if api_key and not api_key.startswith("your_"):
        try:
            from ibm_watson import LanguageTranslatorV3
            from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

            authenticator = IAMAuthenticator(api_key)
            lt = LanguageTranslatorV3(version="2018-05-01", authenticator=authenticator)
            lt.set_service_url(settings.ibm_translator_url)

            model_id = (
                f"{source_lang}-{target_lang}"
                if source_lang != "auto"
                else None
            )
            result = lt.translate(
                text=[text],
                model_id=model_id,
                target=target_lang if source_lang == "auto" else None,
            ).get_result()
            translation = result["translations"][0]["translation"]
            logger.debug("IBM Watson translated to %s: %.50s", target_lang, translation)
            return translation
        except Exception as e:
            logger.warning("IBM Watson Translator failed: %s — falling back.", e)

    # Fallback: deep-translator
    try:
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source="auto", target=target_lang)
        translated = translator.translate(text)
        return translated or text
    except Exception as e:
        logger.error("deep-translator fallback failed: %s", e)
        return text


def translate_to_english(text: str) -> tuple[str, str]:
    """
    Translate text to English if it is not already English.
    Returns (translated_text, detected_language_code).
    """
    lang = detect_language(text)
    if lang == "en":
        return text, "en"
    translated = translate_text(text, target_lang="en", source_lang=lang)
    return translated, lang


def translate_response(response: str, target_lang: str) -> str:
    """Translate an English response to the farmer's preferred language."""
    if target_lang == "en":
        return response
    return translate_text(response, target_lang=target_lang, source_lang="en")
