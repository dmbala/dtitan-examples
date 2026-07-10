# Workshop configs

Level 1 runs the **built-in** TorchTitan config `--module llama3 --config
llama3_debugmodel` plus dotted CLI overrides — no custom config registration.
Later levels may register workshop-specific configs in the model's
`config_registry`; that API is confirmed on the rebuilt torch>=2.11 container.
