import json
import os

import numpy as np
import torch


# sdpa-diagnostics
class AttentionNpyCapture:
    """Capture final-layer MHA outputs from the first model evaluation."""

    def __init__(
        self,
        model,
        output_dir,
        *,
        run_mode,
        use_sdpa,
        seed,
        checkpoint=None,
    ):
        self.output_dir = os.path.abspath(output_dir)
        if os.path.exists(self.output_dir) and not os.path.isdir(self.output_dir):
            raise NotADirectoryError(
                f'--attn_to_npy must point to a folder: {self.output_dir}'
            )
        os.makedirs(self.output_dir, exist_ok=True)

        latent_model = model.model
        if len(latent_model.layers) == 0:
            raise ValueError('Cannot capture attention: the model has no MDGen layers')

        final_layer_index = len(latent_model.layers) - 1
        final_layer = latent_model.layers[final_layer_index]
        if not hasattr(final_layer.mha_t, 'attn'):
            raise ValueError(
                '--attn_to_npy requires frame-axis MHA, but this model uses '
                'a non-MHA frame operator (for example Hyena)'
            )

        targets = {
            'residue': (
                final_layer.mha_l.attn,
                f'model.layers.{final_layer_index}.mha_l.attn',
            ),
            'frame': (
                final_layer.mha_t.attn,
                f'model.layers.{final_layer_index}.mha_t.attn',
            ),
        }

        ipa_layers = getattr(latent_model, 'ipa_layers', None)
        if ipa_layers is not None and len(ipa_layers) > 0:
            final_ipa_index = len(ipa_layers) - 1
            targets['ipa'] = (
                ipa_layers[final_ipa_index].mha_l.attn,
                f'model.ipa_layers.{final_ipa_index}.mha_l.attn',
            )

        self.target_paths = {
            label: os.path.join(self.output_dir, f'{label}.npy')
            for label in targets
        }
        self.metadata_path = os.path.join(self.output_dir, 'metadata.json')
        existing_paths = [
            path for path in self.target_paths.values() if os.path.exists(path)
        ]
        if os.path.exists(self.metadata_path):
            existing_paths.append(self.metadata_path)
        if existing_paths:
            raise FileExistsError(
                '--attn_to_npy will not overwrite existing capture files: '
                + ', '.join(existing_paths)
            )

        checkpoint_path = None
        if checkpoint is not None:
            checkpoint_path = os.path.abspath(checkpoint)
        self.captured = set()
        self.metadata = {
            'run_mode': run_mode,
            'requested_backend': 'sdpa' if use_sdpa else 'manual',
            'checkpoint': checkpoint_path,
            'seed': seed,
            'capture_scope': (
                f'first {run_mode} model evaluation, final layer only'
            ),
            'tensor_boundary': 'MultiheadAttention output after out_proj',
            'sample_name': None,
            'prepend_ipa_available': 'ipa' in targets,
            'captures': {},
        }
        self._write_metadata()

        self.handles = []
        for label, (module, module_name) in targets.items():
            self.handles.append(
                module.register_forward_hook(self._make_hook(label, module_name))
            )

        if 'ipa' not in targets:
            print(
                'Attention NPY capture: prepend IPA is not enabled; '
                'ipa.npy will not be created.'
            )

    def set_sample_name(self, name):
        if not self.captured and self.metadata['sample_name'] is None:
            self.metadata['sample_name'] = name
            self._write_metadata()

    def _make_hook(self, label, module_name):
        def save_output(module, inputs, output):
            if label in self.captured:
                return
            attn = output[0] if isinstance(output, tuple) else output
            if not isinstance(attn, torch.Tensor):
                raise TypeError(
                    f'{module_name} returned a non-tensor attention output'
                )

            array = attn.detach().cpu().numpy()
            np.save(self.target_paths[label], array, allow_pickle=False)
            self.captured.add(label)
            self.metadata['captures'][label] = {
                'module': module_name,
                'shape': list(array.shape),
                'dtype': str(array.dtype),
                'actual_backend': getattr(
                    module, 'last_attention_backend', 'unknown'
                ),
                'file': os.path.basename(self.target_paths[label]),
            }
            self._write_metadata()
            print(
                f'Attention NPY saved: {self.target_paths[label]} '
                f'(shape={array.shape})'
            )

        return save_output

    def _write_metadata(self):
        with open(self.metadata_path, 'w', encoding='utf-8') as handle:
            json.dump(self.metadata, handle, indent=2, sort_keys=True)
            handle.write('\n')
