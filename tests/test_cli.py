import subprocess


def test_cli_info():
    result = subprocess.run(["python", "-m", "oimalib", "--info"], capture_output=True)
    assert "Demo" in result.stdout.decode()
