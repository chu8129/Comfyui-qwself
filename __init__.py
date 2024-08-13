import torch

if "AMD" in torch.cuda.get_device_name() or "Radeon" in torch.cuda.get_device_name() or "NVIDIA" in torch.cuda.get_device_name():
    try:
        from flash_attn import flash_attn_func

        sdpa = torch.nn.functional.scaled_dot_product_attention

        def sdpa_hijack(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
        ):
            if query.shape[3] <= 128 and attn_mask is None and query.dtype != torch.float32:
                hidden_states = flash_attn_func(
                    q=query.transpose(1, 2),
                    k=key.transpose(1, 2),
                    v=value.transpose(1, 2),
                    dropout_p=dropout_p,
                    causal=is_causal,
                    softmax_scale=scale,
                ).transpose(1, 2)
            else:
                hidden_states = sdpa(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )
            return hidden_states

        torch.nn.functional.scaled_dot_product_attention = sdpa_hijack
        print("# # #\nAMD GO FAST\n# # #")
    except ImportError as e:
        print(f"# # #\nAMD GO SLOW\n{e}\n# # #")
else:
    print(f"# # #\nAMD GO SLOW\nCould not detect AMD GPU from:\n{torch.cuda.get_device_name()}\n# # #")


import importlib
import logging as log
import traceback

from pathlib import Path
import sys
import os

"""
libsPath = os.path.abspath(os.path.join(os.path.dirname(__file__), "codes"))
print(libsPath)
sys.path.append(libsPath)
"""


def load_nodes():
    shorted_errors = []
    full_error_messages = []
    node_class_mappings = {}
    node_display_name_mappings = {}

    here = Path(__file__).parent.resolve()
    # sys.path.append(str(here)/ "codes")

    for filename in (here / "nodes").iterdir():
        module_name = filename.stem

        if module_name.endswith(".py"):
            continue

        try:
            module = importlib.import_module(f".nodes.{module_name}", package=__package__)
            node_class_mappings.update(getattr(module, "NODE_CLASS_MAPPINGS"))
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                node_display_name_mappings.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS"))

            print(f"Imported {module_name} nodes")

        except AttributeError:
            pass  # wip nodes
            print("ignore this pass")
        except Exception:
            error_message = traceback.format_exc()
            full_error_messages.append(error_message)
            error_message = error_message.splitlines()[-1]
            shorted_errors.append(f"Failed to import module {module_name} because {error_message}")

    if len(shorted_errors) > 0:
        full_err_log = "\n\n".join(full_error_messages)
        print(f"\n\nFull error log from comfyui_controlnet_aux: \n{full_err_log}\n\n")
        log.info(
            f"Some nodes failed to load:\n\t"
            + "\n\t".join(shorted_errors)
            + "\n\n"
            + "Check that you properly installed the dependencies.\n"
            + "If you think this is a bug, please report it on the github page (https://github.com/Fannovel16/comfyui_controlnet_aux/issues)"
        )
    return node_class_mappings, node_display_name_mappings


NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = load_nodes()
