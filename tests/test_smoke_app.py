# tests/test_smoke_app.py
import pytest
import importlib

@pytest.mark.parametrize("module_name", [
    "app.utils.iteratia1",
    "app.utils.iteratia2",
    "app.utils.iteratia3"
])
def test_utils_modules_import(module_name):
    """
    Verificăm că modulele din app.utils se importă fără erori.
    """
    try:
        importlib.import_module(module_name)
    except Exception as e:
        pytest.fail(f"Import modul '{module_name}' eșuat: {e}")

def test_app_imports_without_errors():
    """
    Verificăm că app.py se importă fără erori (smoke test).
    """
    try:
        import app  # va arunca excepție dacă există ceva greșit
    except Exception as e:
        pytest.fail(f"Importul lui app.py a eșuat cu excepția: {e}")
