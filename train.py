from mdgen.parsing import parse_train_args
args = parse_train_args()
use_sdpa = args.use_sdpa  # sdpa-route
print_sdpa_backend = args.print_sdpa_backend  # sdpa-diagnostics
attn_to_npy = args.attn_to_npy  # sdpa-diagnostics
delattr(args, 'use_sdpa')  # sdpa-route
delattr(args, 'print_sdpa_backend')  # sdpa-diagnostics
delattr(args, 'attn_to_npy')  # sdpa-diagnostics
from mdgen.logger import get_logger
logger = get_logger(__name__)

import torch, os, wandb
from mdgen.dataset import MDGenDataset
from mdgen.attn_capture import AttentionNpyCapture  # sdpa-diagnostics
from mdgen.wrapper import NewMDGenWrapper
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
import pytorch_lightning as pl


# sdpa-diagnostics
def get_sdpa_backend(operator_name):
    backend_operators = {
        '_scaled_dot_product_flash_attention': 'FLASH_ATTENTION',
        '_scaled_dot_product_efficient_attention': 'EFFICIENT_ATTENTION',
        '_scaled_dot_product_cudnn_attention': 'CUDNN_ATTENTION',
        '_scaled_dot_product_attention_math': 'MATH',
    }
    for operator, backend in backend_operators.items():
        if operator in operator_name:
            return backend
    return None


# sdpa-diagnostics
class SDPABackendReportCallback(pl.Callback):
    def __init__(self, output_path, use_sdpa):
        super().__init__()
        self.output_path = os.path.abspath(output_path)
        self.use_sdpa = use_sdpa
        self.profiler = None
        self.reported = False
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.reported:
            return
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        self.profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
        )
        self.profiler.start()

    def on_before_backward(self, trainer, pl_module, loss):
        if self.reported or self.profiler is None:
            return
        self.profiler.stop()
        self._write_report(pl_module)
        self.profiler = None
        self.reported = True
        print(f'SDPA backend report saved to: {self.output_path}')

    def _write_report(self, model):
        lines = [
            'SDPA backend report',
            'mode: training forward',
            f'requested_attention_path: {"sdpa" if self.use_sdpa else "manual"}',
            f'torch_version: {torch.__version__}',
            f'cuda_wheel: {torch.version.cuda}',
            f'gpu: {torch.cuda.get_device_name(0)}',
            f'lightning_precision: {args.precision}',
            '',
            'Enabled PyTorch SDPA backends:',
        ]
        enabled_checks = (
            ('FLASH_ATTENTION', 'flash_sdp_enabled'),
            ('EFFICIENT_ATTENTION', 'mem_efficient_sdp_enabled'),
            ('CUDNN_ATTENTION', 'cudnn_sdp_enabled'),
            ('MATH', 'math_sdp_enabled'),
        )
        for backend, check_name in enabled_checks:
            check = getattr(torch.backends.cuda, check_name, None)
            lines.append(f'{backend}: {check() if check is not None else "unknown"}')

        lines.extend(['', 'MDGen attention modules:'])
        found_module = False
        for name, module in model.named_modules():
            if not hasattr(module, 'last_attention_backend'):
                continue
            found_module = True
            backend = module.last_attention_backend or 'not_executed'
            fallback = module.last_sdpa_fallback_reason
            message = f'{name}: path={backend}'
            if fallback is not None:
                message += f', fallback={fallback}'
            lines.append(message)
        if not found_module:
            lines.append('No attention modules expose backend information.')

        lines.extend(['', 'PyTorch SDPA operators observed:'])
        observed_backends = set()
        observed_operators = []
        events = self.profiler.key_averages(group_by_input_shape=True)
        for event in events:
            if 'scaled_dot_product' not in event.key:
                continue
            backend = get_sdpa_backend(event.key)
            if backend is not None:
                observed_backends.add(backend)
            observed_operators.append((event, backend))

        if observed_backends:
            lines.append(f'selected_backends: {", ".join(sorted(observed_backends))}')
        else:
            lines.append('selected_backends: none detected')

        if not observed_operators:
            lines.append('No scaled-dot-product attention operators were observed.')
        for event, backend in observed_operators:
            self_device_time = getattr(
                event,
                'self_device_time_total',
                getattr(event, 'self_cuda_time_total', 0.0),
            )
            device_time = getattr(
                event,
                'device_time_total',
                getattr(event, 'cuda_time_total', 0.0),
            )
            lines.extend([
                f'operator: {event.key}',
                f'  backend: {backend or "dispatcher/unknown"}',
                f'  calls: {event.count}',
                f'  input_shapes: {event.input_shapes}',
                f'  self_cpu_time_total_us: {event.self_cpu_time_total}',
                f'  cpu_time_total_us: {event.cpu_time_total}',
                f'  self_device_time_total_us: {self_device_time}',
                f'  device_time_total_us: {device_time}',
            ])

        with open(self.output_path, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(lines) + '\n')


pl.seed_everything(args.train_seed, workers=True)  # seed-initialization

torch.set_float32_matmul_precision('highest')  # fp32-matmul-precision
# fp32-disable-tf32:start
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# fp32-disable-tf32:end

if args.wandb:
    wandb.init(
        entity=os.environ["WANDB_ENTITY"],
        settings=wandb.Settings(start_method="fork"),
        project="mdgen",
        name=args.run_name,
        config=args,
    )


trainset = MDGenDataset(args, split=args.train_split)

if args.overfit:
    valset = trainset    
else:
    valset = MDGenDataset(args, split=args.val_split, repeat=args.val_repeat)

train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
)

val_loader = torch.utils.data.DataLoader(
    valset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
)
model = NewMDGenWrapper(args, use_sdpa=use_sdpa)  # sdpa-route
# sdpa-diagnostics:start
if attn_to_npy is not None:
    attn_capture = AttentionNpyCapture(
        model,
        attn_to_npy,
        run_mode='validation' if args.validate else 'training',
        use_sdpa=use_sdpa,
        seed=args.train_seed,
        checkpoint=args.ckpt,
    )
    model._attn_npy_capture = attn_capture
# sdpa-diagnostics:end

# sdpa-diagnostics:start
callbacks = [
    ModelCheckpoint(
        dirpath=os.environ["MODEL_DIR"],
        save_top_k=-1,
        every_n_epochs=args.ckpt_freq,
    ),
    ModelSummary(max_depth=2),
]
if print_sdpa_backend is not None:
    callbacks.append(SDPABackendReportCallback(print_sdpa_backend, use_sdpa))
# sdpa-diagnostics:end

trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else 'auto',
    deterministic=args.deterministic,  # deterministic-execution
    benchmark=args.benchmark,  # deterministic-benchmark-guard
    max_epochs=args.epochs,
    limit_train_batches=args.train_batches or 1.0,
    limit_val_batches=0.0 if args.no_validate else (args.val_batches or 1.0),
    num_sanity_val_steps=0,
    precision=args.precision,
    enable_progress_bar=not args.wandb or os.getlogin() == 'hstark',
    gradient_clip_val=args.grad_clip,
    default_root_dir=os.environ["MODEL_DIR"], 
    callbacks=callbacks, # sdpa-diagnostics
    accumulate_grad_batches=args.accumulate_grad,
    val_check_interval=args.val_freq,
    check_val_every_n_epoch=args.val_epoch_freq,
    logger=False
)

if args.validate:
    trainer.validate(model, val_loader, ckpt_path=args.ckpt)
else:
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)
