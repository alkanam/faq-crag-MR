from pathlib import Path
import os


def load_dotenv(dotenv_path):
    path = Path(dotenv_path)
    if not path.exists():
        return False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if len(value) >= 2 and ((value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'"))):
            value = value[1:-1]

        if key not in os.environ:
            os.environ[key] = value

    return True


load_dotenv(Path(__file__).parent / ".env")
