import modal

app = modal.App("qwen35-vl-pipeline")

vol = modal.Volume.from_name("qwen35-vl", create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers>=5.2.0",
        "onnx",
        "onnxruntime",
        "onnxscript",
        "optimum",
        "huggingface_hub",
        "qai-hub",
        "numpy",
        "Pillow",
        "accelerate",
    )
)

@app.function(
    image=base_image,
    volumes={"/cache": vol},
    timeout=1800,
)
def download_weights(hf_model_id: str):
    from huggingface_hub import snapshot_download
    snapshot_download(
        hf_model_id,
        local_dir="/cache/weights",
    )
    vol.commit()
    print("✓ Weights saved to /cache/weights")


@app.function(
    gpu="A100-40GB",
    image=base_image,
    volumes={"/cache": vol},
    timeout=7200,
)
def export_full_model_to_onnx(prompt_length: int = 128, image_size: int = 224):
    vol.reload()

    import os
    import json
    import torch
    import numpy as np
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    out_dir = "/cache/onnx_full"
    os.makedirs(out_dir, exist_ok=True)

    for filename in [
        "full_model_fp32.onnx",
        "input_specs.json",
        "sample_inputs.npz",
    ]:
        path = os.path.join(out_dir, filename)
        if os.path.exists(path):
            os.remove(path)

    print("Loading model in FP32...")
    model = AutoModelForImageTextToText.from_pretrained(
        "/cache/weights",
        dtype=torch.float32,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        "/cache/weights",
        trust_remote_code=True,
    )

    class FullVLMWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(
            self,
            input_ids,
            attention_mask,
            mm_token_type_ids,
            pixel_values,
            image_grid_thw,
        ):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
            return outputs.logits

    wrapped = FullVLMWrapper(model).eval()

    dummy_image = Image.fromarray(
        np.zeros((image_size, image_size, 3), dtype=np.uint8)
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": dummy_image},
                {"type": "text", "text": "a " * prompt_length},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[dummy_image],
        return_tensors="pt",
    ).to("cuda")

    required = [
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "pixel_values",
        "image_grid_thw",
    ]

    for key in required:
        if key not in inputs:
            raise KeyError(f"Processor did not return required input: {key}")

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    mm_token_type_ids = inputs["mm_token_type_ids"]
    pixel_values = inputs["pixel_values"].float()
    image_grid_thw = inputs["image_grid_thw"]

    print("ONNX export inputs:")
    tensors = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_token_type_ids,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }

    for name, tensor in tensors.items():
        print(name, tuple(tensor.shape), tensor.dtype)

    with torch.no_grad():
        torch_out = wrapped(
            input_ids,
            attention_mask,
            mm_token_type_ids,
            pixel_values,
            image_grid_thw,
        )
        print("PyTorch output:", tuple(torch_out.shape), torch_out.dtype)

    onnx_path = f"{out_dir}/full_model_fp32.onnx"

    print("Exporting ONNX...")
    torch.onnx.export(
        wrapped,
        args=(
            input_ids,
            attention_mask,
            mm_token_type_ids,
            pixel_values,
            image_grid_thw,
        ),
        f=onnx_path,
        input_names=[
            "input_ids",
            "attention_mask",
            "mm_token_type_ids",
            "pixel_values",
            "image_grid_thw",
        ],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )

    input_specs = {
        "input_ids": [list(input_ids.shape), "int64"],
        "attention_mask": [list(attention_mask.shape), "int64"],
        "mm_token_type_ids": [list(mm_token_type_ids.shape), "int64"],
        "pixel_values": [list(pixel_values.shape), "float32"],
        "image_grid_thw": [list(image_grid_thw.shape), "int64"],
    }

    with open(f"{out_dir}/input_specs.json", "w") as f:
        json.dump(input_specs, f, indent=2)

    np.savez(
        f"{out_dir}/sample_inputs.npz",
        input_ids=input_ids.cpu().numpy(),
        attention_mask=attention_mask.cpu().numpy(),
        mm_token_type_ids=mm_token_type_ids.cpu().numpy(),
        pixel_values=pixel_values.cpu().numpy(),
        image_grid_thw=image_grid_thw.cpu().numpy(),
    )

    file_size_gb = os.path.getsize(onnx_path) / (1024 ** 3)
    print(f"✓ Saved ONNX: {onnx_path}")
    print(f"✓ ONNX size: {file_size_gb:.2f} GB")
    print(f"✓ Saved input specs and sample inputs to {out_dir}")

    vol.commit()

@app.function(
    image=base_image,
    volumes={"/cache": vol},
    timeout=1800,
)
def check_full_model_onnx():
    vol.reload()

    import os
    import onnx

    onnx_path = "/cache/onnx_full/full_model_fp32.onnx"

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    size_gb = os.path.getsize(onnx_path) / (1024 ** 3)
    print(f"ONNX size: {size_gb:.2f} GB")

    print("Checking ONNX model by path...")
    onnx.checker.check_model(onnx_path)

    print("✓ ONNX checker passed.")

@app.function(
    image=base_image,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("qai-hub-secret")],
    timeout=7200,
)
def compile_full_onnx_fp32_qnn():
    vol.reload()

    import os
    import json
    import shutil
    import onnx
    import onnxruntime as ort
    import qai_hub as hub

    token = os.environ.get("QAI_HUB_API_TOKEN")
    client = hub.Client(config=hub.ClientConfig(api_token=token))

    onnx_dir = "/cache/onnx_full"
    upload_dir = "/cache/onnx_full_upload"

    onnx_path = f"{onnx_dir}/full_model_fp32.onnx"
    specs_path = f"{onnx_dir}/input_specs.json"

    if not os.path.isdir(onnx_dir):
        raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}")

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    if not os.path.exists(specs_path):
        raise FileNotFoundError(f"Input specs file not found: {specs_path}")

    print("Original ONNX directory contents:")
    for name in os.listdir(onnx_dir):
        path = os.path.join(onnx_dir, name)
        if os.path.isfile(path):
            size_gb = os.path.getsize(path) / (1024 ** 3)
            print(name, f"{size_gb:.3f} GB")
        else:
            print(name, "<DIR>")

    print("Reading actual ONNX inputs from ONNX Runtime...")
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    actual_input_names = [inp.name for inp in session.get_inputs()]

    print("Actual ONNX inputs:")
    for inp in session.get_inputs():
        print(inp.name, inp.shape, inp.type)

    with open(specs_path, "r") as f:
        raw_specs = json.load(f)

    all_specs = {
        name: (tuple(shape_dtype[0]), shape_dtype[1])
        for name, shape_dtype in raw_specs.items()
    }

    input_specs = {
        name: all_specs[name]
        for name in actual_input_names
        if name in all_specs
    }

    missing_specs = set(actual_input_names) - set(input_specs.keys())

    if missing_specs:
        raise ValueError(f"Missing specs for ONNX inputs: {missing_specs}")

    print("Filtered AI Hub input specs:")
    for k, v in input_specs.items():
        print(k, v)

    # Build a clean ONNX model directory for AI Hub.
    # AI Hub allows only .onnx, .data, .encodings, and .bin files.
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)

    os.makedirs(upload_dir, exist_ok=True)

    import onnx
    from onnx.external_data_helper import load_external_data_for_model

    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)

    os.makedirs(upload_dir, exist_ok=True)

    print("Loading ONNX metadata without external data...")
    model = onnx.load_model(onnx_path, load_external_data=False)

    print("Loading external data from original ONNX directory...")
    load_external_data_for_model(model, onnx_dir)

    sanitized_onnx_path = f"{upload_dir}/full_model_fp32.onnx"

    print("Saving sanitized ONNX with one external .data file...")
    onnx.save_model(
        model,
        sanitized_onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=4096,
        convert_attribute=True,
    )

    print("Clean upload directory contents:")
    for name in os.listdir(upload_dir):
        path = os.path.join(upload_dir, name)
        size_gb = os.path.getsize(path) / (1024 ** 3)
        print(name, f"{size_gb:.3f} GB")

    print("Checking sanitized ONNX by path...")
    onnx.checker.check_model(sanitized_onnx_path)
    print("✓ Sanitized ONNX checker passed.")

    print("Uploading clean ONNX model directory to AI Hub...")
    uploaded_model = client.upload_model(upload_dir)
    print(f"Uploaded model ID: {uploaded_model.model_id}")

    device = hub.Device("QCS8550 (Proxy)")

    print("Submitting FP32 ONNX QNN compile job...")
    compile_job = client.submit_compile_job(
        model=uploaded_model,
        device=device,
        name="qwen-vl-full-onnx-fp32-qnn",
        input_specs=input_specs,
        options="--target_runtime onnx",
    )

    print(f"Compile job: {compile_job.url}")

    status = compile_job.wait()

    print(f"Compile status: {status.code}")
    print(f"Compile message: {status.message}")

    if "FAILED" in str(status.code):
        print(f"Compile failed. Check: {compile_job.url}")
        return

    target_model = compile_job.get_target_model()

    with open(f"{onnx_dir}/compiled_fp32_model_id.txt", "w") as f:
        f.write(target_model.model_id)

    vol.commit()

    print(f"✓ FP32 QNN compiled model ID: {target_model.model_id}")
    print(f"✓ Compile job: {compile_job.url}")


@app.function(
    image=base_image,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("qai-hub-secret")],
    timeout=7200,
)
def profile_compiled_onnx_model(model_id: str):
    vol.reload()

    import os
    import qai_hub as hub

    token = os.environ.get("QAI_HUB_API_TOKEN")
    client = hub.Client(config=hub.ClientConfig(api_token=token))

    print(f"Loading compiled model from AI Hub: {model_id}")
    compiled_model = client.get_model(model_id)

    device = hub.Device("QCS8550 (Proxy)")

    print("Submitting profile job...")
    profile_job = client.submit_profile_job(
        model=compiled_model,
        device=device,
        name="qwen-vl-full-onnx-runtime-profile",
    )

    print(f"Profile job: {profile_job.url}")

    status = profile_job.wait()

    print(f"Profile status: {status.code}")
    print(f"Profile message: {status.message}")

    if "FAILED" in str(status.code):
        print(f"Profile failed. Check: {profile_job.url}")
        return

    print(f"✓ Profile complete: {profile_job.url}")

    try:
        profile = profile_job.download_profile()
        print("Downloaded profile:")
        print(profile)
    except Exception as e:
        print("Could not download profile directly.")
        print(e)
        print(f"Open profile manually: {profile_job.url}")

# Quantization image — adds AIMET (Linux only, hence Modal)
quant_image = base_image.pip_install("aimet-onnx")