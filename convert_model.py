import h5py
import json
import tensorflow as tf

old_path = "models/model.h5"
new_path = "models/model_fixed.keras"

print("📌 Loading raw H5 model file...")
with h5py.File(old_path, "r") as f:
    raw = f.attrs["model_config"]

    # Handle bytes/string
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    config = json.loads(raw)

# === Full sanitization function =============================
def clean_layer(layer):
    cfg = layer.get("config", {})

    # Remove dtype policies fully
    if "dtype" in cfg:
        if isinstance(cfg["dtype"], dict):
            # OLD STYLE POLICY → REPLACE WITH PLAIN STRING
            cfg["dtype"] = "float32"
        elif isinstance(cfg["dtype"], str):
            cfg["dtype"] = "float32"

    # Remove batch input shapes
    for key in ["batch_shape", "batch_input_shape"]:
        if key in cfg:
            cfg.pop(key)

    # InputLayer cleanup
    if layer.get("class_name") == "InputLayer":
        for key in ["sparse", "ragged"]:
            cfg.pop(key, None)

    layer["config"] = cfg
    return layer

# Clean every layer
print("🔧 Removing old dtype/batch policies from layers...")
config["config"]["layers"] = [clean_layer(l) for l in config["config"]["layers"]]

# Clean model-level dtype entries
if "dtype" in config["config"]:
    config["config"]["dtype"] = "float32"

# ===============================================================

print("🛠 Rebuilding sanitized model...")
model = tf.keras.models.model_from_config(config)

print("💾 Saving in TF 2.x format...")
model.save(new_path)

print("\n🎉 DONE! Converted model saved as:", new_path)
