import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument('--sim_ckpt', type=str, default=None, required=True)
parser.add_argument('--data_dir', type=str, default=None, required=True)
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--pdb_id', nargs='*', default=[])
parser.add_argument('--num_frames', type=int, default=1000)
parser.add_argument('--num_rollouts', type=int, default=100)
parser.add_argument('--no_frames', action='store_true')
parser.add_argument('--tps', action='store_true')
parser.add_argument('--xtc', action='store_true')
parser.add_argument('--out_dir', type=str, default=".")
parser.add_argument('--split', type=str, default='splits/4AA_test.csv')
# seed-deterministic-args
parser.add_argument('--inference_seed', type=int, default=137) 
parser.add_argument('--use_sdpa', action='store_true')
parser.add_argument('--print_sdpa_backend', type=str, default=None, metavar='PATH')
parser.add_argument(
    '--attn_to_npy',
    type=str,
    default=None,
    metavar='FOLDER_PATH',
    help=(
        'Save the first inference evaluation from the final residue, frame, '
        'and prepend-IPA MHA layers as NPY files.'
    ),
)
parser.add_argument(
    '--deterministic',
    action=argparse.BooleanOptionalAction,
    default=True,
    help='Use deterministic algorithms when available.',
)
parser.add_argument(
    '--benchmark',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Enable the cuDNN benchmark autotuner.',
)
args = parser.parse_args()
# deterministic-benchmark-guard
if args.deterministic and args.benchmark:
    parser.error('--deterministic and --benchmark cannot both be enabled')

import os, torch, mdtraj, tqdm, time
import numpy as np
from pytorch_lightning import seed_everything 
from mdgen.geometry import atom14_to_frames, atom14_to_atom37, atom37_to_torsions
from mdgen.attn_capture import AttentionNpyCapture
from mdgen.residue_constants import restype_order, restype_atom37_mask
from mdgen.tensor_utils import tensor_tree_map
from mdgen.wrapper import NewMDGenWrapper
from mdgen.utils import atom14_to_pdb
import pandas as pd

# seed-initialization
seed_everything(args.inference_seed, workers=True) 
# deterministic-execution
torch.use_deterministic_algorithms(args.deterministic)
torch.backends.cudnn.benchmark = args.benchmark
# fp32-matmul-precision
# fp32-disable-tf32
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


os.makedirs(args.out_dir, exist_ok=True)


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


def write_sdpa_backend_report(model, profiler):
    output_path = os.path.abspath(args.print_sdpa_backend)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    lines = [
        'SDPA backend report',
        f'requested_attention_path: {"sdpa" if args.use_sdpa else "manual"}',
        f'torch_version: {torch.__version__}',
        f'cuda_wheel: {torch.version.cuda}',
        f'gpu: {torch.cuda.get_device_name(0)}',
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
    events = profiler.key_averages(group_by_input_shape=True)
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

    with open(output_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    model._sdpa_backend_reported = True
    print(f'SDPA backend report saved to: {output_path}')


def get_batch(name, seqres, position_ids, num_frames):
    arr = np.lib.format.open_memmap(f'{args.data_dir}/{name}{args.suffix}.npy', 'r')

    if not args.tps: # else keep all frames
        arr = arr[0:1]
    arr = np.array(arr, dtype=np.float32, copy=True)

    position_ids = torch.as_tensor(position_ids, dtype=torch.long)
    if position_ids.ndim != 1 or len(position_ids) != len(seqres):
        raise ValueError(
            f'{name}: seqres length {len(seqres)} does not match '
            f'position_ids shape {tuple(position_ids.shape)}'
        )
    if arr.shape[1] != len(seqres):
        raise ValueError(
            f'{name}: NPY residue count {arr.shape[1]} does not match '
            f'CSV seqres/position_ids length {len(seqres)}'
        )

    frames = atom14_to_frames(torch.from_numpy(arr))
    seqres = torch.tensor([restype_order[c] for c in seqres])
    atom37 = torch.from_numpy(atom14_to_atom37(arr, seqres[None])).float()
    L = len(seqres)
    mask = torch.ones(L)
    
    if args.no_frames:
        return {
            'atom37': atom37,
            'seqres': seqres,
            'position_ids': position_ids,
            'mask': restype_atom37_mask[seqres],
        }
        
    torsions, torsion_mask = atom37_to_torsions(atom37, seqres[None])
    return {
        'torsions': torsions,
        'torsion_mask': torsion_mask[0],
        'trans': frames._trans,
        'rots': frames._rots._rot_mats,
        'seqres': seqres,
        'position_ids': position_ids,
        'mask': mask, # (L,)
    }

def rollout(model, batch):

    #print('Start sim', batch['trans'][0,0,0])
    if args.no_frames:
        
        expanded_batch = {
            'atom37': batch['atom37'].expand(-1, args.num_frames, -1, -1, -1),
            'seqres': batch['seqres'],
            'position_ids': batch['position_ids'],
            'mask': batch['mask'],
        }
    else:    
        expanded_batch = {
            'torsions': batch['torsions'].expand(-1, args.num_frames, -1, -1, -1),
            'torsion_mask': batch['torsion_mask'],
            'trans': batch['trans'].expand(-1, args.num_frames, -1, -1),
            'rots': batch['rots'].expand(-1, args.num_frames, -1, -1, -1),
            'seqres': batch['seqres'],
            'position_ids': batch['position_ids'],
            'mask': batch['mask'],
        }
    should_profile = (
        args.print_sdpa_backend is not None
        and not getattr(model, '_sdpa_backend_reported', False)
    )
    if should_profile:
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
            atom14, _ = model.inference(expanded_batch)
        write_sdpa_backend_report(model, profiler)
    else:
        atom14, _ = model.inference(expanded_batch)
    new_batch = {**batch}

    if args.no_frames:
        new_batch['atom37'] = torch.from_numpy(
            atom14_to_atom37(atom14[:,-1].cpu(), batch['seqres'][0].cpu())
        ).cuda()[:,None].float()
        
        
        
    else:
        frames = atom14_to_frames(atom14[:,-1])
        new_batch['trans'] = frames._trans[None]
        new_batch['rots'] = frames._rots._rot_mats[None]
        atom37 = atom14_to_atom37(atom14[0,-1].cpu(), batch['seqres'][0].cpu())
        torsions, _ = atom37_to_torsions(atom37, batch['seqres'][0].cpu())
        new_batch['torsions'] = torsions[None, None].cuda()

    return atom14, new_batch
    
    
def do(model, name, seqres, position_ids):

    item = get_batch(
        name, seqres, position_ids, num_frames=model.args.num_frames
    )
    batch = next(iter(torch.utils.data.DataLoader([item])))

    batch = tensor_tree_map(lambda x: x.cuda(), batch)  
    
    all_atom14 = []
    start = time.time()
    for _ in tqdm.trange(args.num_rollouts):
        atom14, batch = rollout(model, batch)
        # print(atom14[0,0,0,1], atom14[0,-1,0,1])
        all_atom14.append(atom14)

    print(time.time() - start)
    all_atom14 = torch.cat(all_atom14, 1)
    
    path = os.path.join(args.out_dir, f'{name}.pdb')
    atom14_to_pdb(all_atom14[0].cpu().numpy(), batch['seqres'][0].cpu().numpy(), path)

    if args.xtc:
        traj = mdtraj.load(path)
        traj.superpose(traj)
        traj.save(os.path.join(args.out_dir, f'{name}.xtc'))
        traj[0].save(os.path.join(args.out_dir, f'{name}.pdb'))

@torch.no_grad()
def main():
    model = NewMDGenWrapper.load_from_checkpoint(
        args.sim_ckpt,
        use_sdpa=args.use_sdpa,
        weights_only=False,
    )
    # fp32-inference-model
    model.eval().float().to('cuda')

    attn_capture = None
    if args.attn_to_npy is not None:
        attn_capture = AttentionNpyCapture(
            model,
            args.attn_to_npy,
            run_mode='inference',
            use_sdpa=args.use_sdpa,
            seed=args.inference_seed,
            checkpoint=args.sim_ckpt,
        )
        model._attn_npy_capture = attn_capture
    
    df = pd.read_csv(args.split, index_col='name')
    for name in df.index:
        if args.pdb_id and name not in args.pdb_id:
            continue
        seqres = df.seqres[name]
        # position-id-inference-input
        if 'position_ids' in df.columns:
            position_ids = json.loads(df.position_ids[name])
        else:
            position_ids = list(range(len(seqres)))
        if attn_capture is not None:
            attn_capture.set_sample_name(name)
        do(model, name, seqres, position_ids)
        

main()
