"""Verify that every project dependency imports correctly."""

packages = [
    "transformers",
    "peft",
    "bitsandbytes",
    "accelerate",
    "datasets",
    "trl",
    "openai",
    "pypdf",
    "dotenv",
]

failures = []

for name in packages:
    try:
        __import__(name)
        print(f"OK   {name}")
    except Exception as e:
        failures.append((name, e))
        print(f"FAIL {name}: {e}")

if failures:
    print(f"\n{len(failures)} package(s) failed to import.")
else:
    print("\nAll packages imported successfully.")
