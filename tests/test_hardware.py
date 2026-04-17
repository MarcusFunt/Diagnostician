from diagnostician.setup import hardware


def test_scanner_selects_qwen4b_for_current_laptop_class(monkeypatch):
    def runner(command, timeout):
        joined = " ".join(command)
        if "Win32_ComputerSystem" in joined:
            return '{"TotalPhysicalMemory":16775000064}'
        if "Win32_OperatingSystem" in joined:
            return '{"FreePhysicalMemory":8388608}'
        if "Win32_Processor" in joined:
            return '{"Name":"13th Gen Intel(R) Core(TM) i7-1355U","NumberOfCores":10}'
        if "Win32_VideoController" in joined:
            return '{"Name":"Intel(R) Iris(R) Xe Graphics","AdapterRAM":2147479552}'
        raise RuntimeError("command not available")

    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hardware, "_ollama_reachable", lambda base_url: True)

    scan = hardware.scan_hardware(run_command=runner)
    selection = hardware.select_models(scan)

    assert scan.cpu_cores == 10
    assert scan.max_vram_gb == 2.0
    assert selection.case_generator_model == "qwen3:4b-instruct"
    assert selection.medical_check_enabled is False
    assert selection.ollama_keep_alive == "5m"
    assert any("Med42 checks will be disabled" in warning for warning in selection.warnings)


def test_model_selection_handles_low_ram_and_gpu_tiers():
    low_ram = hardware.select_models(hardware.HardwareScan(total_ram_gb=8))
    gpu_8gb = hardware.select_models(hardware.HardwareScan(total_ram_gb=16, nvidia_vram_gb=[8]))
    gpu_12gb = hardware.select_models(hardware.HardwareScan(total_ram_gb=16, nvidia_vram_gb=[12]))
    large_ram = hardware.select_models(hardware.HardwareScan(total_ram_gb=48))

    assert low_ram.case_generator_model == "qwen3:1.7b"
    assert any("Less than 12 GB RAM" in warning for warning in low_ram.warnings)
    assert gpu_8gb.case_generator_model == "qwen3:8b"
    assert gpu_8gb.medical_check_enabled is True
    assert gpu_12gb.case_generator_model == "qwen3:14b"
    assert large_ram.case_generator_model == "qwen3:30b"


def test_scanner_reports_missing_ollama(monkeypatch):
    def runner(command, timeout):
        joined = " ".join(command)
        if "Win32_ComputerSystem" in joined:
            return '{"TotalPhysicalMemory":8589934592}'
        if "Win32_OperatingSystem" in joined:
            return '{"FreePhysicalMemory":1048576}'
        if "Win32_Processor" in joined:
            return '{"Name":"CPU","NumberOfCores":4}'
        if "Win32_VideoController" in joined:
            return "[]"
        raise RuntimeError("command not available")

    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hardware, "_ollama_reachable", lambda base_url: False)

    scan = hardware.scan_hardware(run_command=runner, ollama_base_url="http://ollama.invalid")

    assert scan.ollama_ok is False
    assert any("Ollama was not reachable" in warning for warning in scan.warnings)


def test_env_file_dry_run_writes_selected_models(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("CUSTOM_VALUE=keep\nDIAGNOSTICIAN_CASE_GENERATOR_MODEL=old\n", encoding="utf-8")
    selection = hardware.ModelSelection(case_generator_model="qwen3:8b", ollama_keep_alive="5m")

    hardware.write_env_file(env_path, hardware.build_env_values(selection))

    text = env_path.read_text(encoding="utf-8")
    assert "CUSTOM_VALUE=keep" in text
    assert "DIAGNOSTICIAN_GENERATION_MODEL=qwen3:8b" in text
    assert "DIAGNOSTICIAN_CASE_GENERATOR_MODEL=qwen3:8b" in text
    assert "DIAGNOSTICIAN_MEDICAL_CHECK_MODEL=hf.co/tensorblock/Llama3-Med42-8B-GGUF" in text
    assert "DIAGNOSTICIAN_MEDICAL_CHECK_ENABLED=true" in text
    assert "DIAGNOSTICIAN_OLLAMA_KEEP_ALIVE=5m" in text
    assert "DIAGNOSTICIAN_LLM_TIMEOUT_SECONDS=600" in text
