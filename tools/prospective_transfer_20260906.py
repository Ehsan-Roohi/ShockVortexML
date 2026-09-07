"""Additive model freeze, exposure audit, and fixed composite inference.

Does not train, invent independent labels, or modify historical artifacts.
"""
from pathlib import Path
import argparse
from collections import Counter
from datetime import datetime, timezone
import numpy as np
import torch
from jcp2026.common import ROOT, read_json, write_json, sha256
from jcp2026.diagnostics import build_inputs
from jcp2026.infer import predict_tiled, clean
from jcp2026.models import make_model
from jcp2026.wake_auxiliary import WakeNet, predict
from ml.flow_aligned import FreestreamReference

CONFIG = ROOT / 'configs/prospective_generalization_20260906.json'
OUT = ROOT / 'results/prospective_generalization_20260906_v1'
SOURCES = [
    'tools/prospective_transfer_20260906.py',
    'src/jcp2026/common.py', 'src/jcp2026/diagnostics.py',
    'src/jcp2026/infer.py', 'src/jcp2026/models.py',
    'src/jcp2026/wake_auxiliary.py', 'src/ml/flow_aligned.py',
    'configs/jcp_wake_auxiliary_20260906_v3.json',
    'configs/jcp_vortex_robustness_20260906_v2.json',
    'tools/show_front_vs_response_20260906.py',
    'tools/f270_expansion_display_20260906.py',
    'configs/f270_expansion_display_20260906.json',
]


def binding(path):
    p = ROOT / path
    return dict(path=str(p.relative_to(ROOT)).replace('\\', '/'), sha256=sha256(p), bytes=p.stat().st_size)


def verify_freeze(path=OUT/'FREEZE.json'):
    freeze = read_json(path)
    for b in freeze['files']:
        if sha256(ROOT/b['path']) != b['sha256']:
            raise ValueError('Frozen artifact changed: '+b['path'])
    return freeze


def eligible(metadata, exposed_cases):
    """Require a pre-receipt registration and explicit exposure declaration.

    Metadata is a declaration, not external proof that no exposure occurred.
    Geometry novelty and condition novelty are kept separate.
    """
    if metadata.get('case_id') in exposed_cases:
        return False
    return (metadata.get('registered_before_field_inspection') is True
            and metadata.get('used_for_training_or_development') is False
            and bool(metadata.get('source_sha256')))


def freeze():
    if (OUT/'FREEZE.json').exists():
        verify_freeze()
        print('Existing freeze verified; not overwritten.')
        return
    cfg = read_json(CONFIG)
    artifacts = [binding(CONFIG), *[binding(p) for p in SOURCES]]
    for b in cfg['primary_model'].values():
        if isinstance(b, dict) and 'checkpoint' in b:
            actual = binding(b['checkpoint'])
            if actual['sha256'] != b['sha256']: raise ValueError('Checkpoint mismatch')
            artifacts.append(actual)
    indexpath = ROOT/'data/processed/jcp_native_primitives_v1/index.json'
    records = read_json(indexpath)['records']
    original = read_json(ROOT/'data/processed/jcp_v4/dataset_index.json')['records']
    original_by_id = {r['dataset_id']: r for r in original}
    groups = Counter((r['case'], original_by_id[r['dataset_id']]['split']) for r in records)
    ledger = [dict(case=case, original_split=split, retained_frames=count,
                   role='exposed_development_or_training', eligible_untouched=False)
              for (case, split), count in sorted(groups.items())]
    audit = dict(native_retained_frames=len(records), cases=ledger,
        other_exposed_families=['SU2 diamond alpha0/4/8/20/40', 'Madani2017 rendered DSMC cylinder', 'hypersonic rarefied cylinder'],
        pending_same_geometry_rerun='SU2 job 64025706; status unknown; do not duplicate',
        genuinely_unseen_geometry_fields_available=False,
        note='Scoped to documented study archives, not a claim to have searched every user disk.',
        comparison_source='results/jcp_v4_corrected_view_v1/comparison_all_seeds_v1.json',
        historical_comparison_independent_accuracy=False)
    write_json(OUT/'EXPOSURE_AUDIT.json', audit)
    artifacts += [binding(indexpath), binding('results/jcp_v4_corrected_view_v1/comparison_all_seeds_v1.json'), binding(OUT/'EXPOSURE_AUDIT.json')]
    write_json(OUT/'FREEZE.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        files=artifacts, config=cfg, validation_status='NOT_YET_EVALUATED_ON_NEW_CFD',
        runtime=dict(torch=torch.__version__, numpy=np.__version__),
        prospective_scope='future registered CFD only; historical development is retrospective'))
    print('FROZEN', len(artifacts), 'artifacts; retained frames', len(records))


class Composite:
    def __init__(self):
        verify_freeze()
        self.cfg = read_json(CONFIG)['primary_model']
        torch.set_num_threads(self.cfg['threads'])
        torch.use_deterministic_algorithms(True)
        self.models = {}
        for name in ('shock', 'vortex_core', 'wake_shear'):
            cp = torch.load(ROOT/self.cfg[name]['checkpoint'], map_location='cpu', weights_only=True)
            model = WakeNet(cp['width']) if name == 'wake_shear' else make_model(cp['spec'], cp['base_channels'])
            model.load_state_dict(cp['model_state_dict']); model.eval()
            self.models[name] = model

    def forward(self, fields, reference):
        reference.validate()
        domain = fields['observation_mask'].astype(bool) & ~fields['geometry'].astype(bool)
        for key in ('rho', 'pressure', 'u', 'v'):
            if fields[key].shape != domain.shape or not np.isfinite(fields[key]).all():
                raise ValueError('Invalid field '+key)
        if not domain.any() or any(np.any(fields[k][domain] <= 0) for k in ('rho', 'pressure')):
            raise ValueError('No valid positive fluid state')
        inputs, _ = build_inputs(fields, reference)
        ans = {}
        for name, head in (('shock', 0), ('vortex_core', 1)):
            p = predict_tiled(self.models[name], torch, inputs,
                tile_size=self.cfg[name]['tile_size'], overlap=self.cfg['overlap'], batch_size=self.cfg['batch_size'])[head]
            ans['p_'+name] = p
            ans[name] = clean((p >= self.cfg[name]['threshold']) & domain, self.cfg['minimum_component_pixels'])
        ans['p_wake_shear'], ans['wake_shear'] = predict(self.models['wake_shear'], inputs, domain,
            self.cfg['wake_shear']['threshold'], self.cfg['minimum_component_pixels'])
        ans['background_other'] = domain & ~(ans['shock'] | ans['vortex_core'] | ans['wake_shear'])
        return ans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze','verify','infer'])
    parser.add_argument('--input', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'freeze': freeze(); return
    verify_freeze()
    if args.action == 'verify': print('FREEZE_OK'); return
    if not all((args.input,args.reference,args.output)): parser.error('infer requires input, reference and output')
    if args.output.exists() or args.output.with_suffix('.json').exists(): raise FileExistsError('Use a new output path')
    with np.load(args.input, allow_pickle=False) as z:
        fields = {k:z[k] for k in ('rho','pressure','u','v','x','y','geometry','observation_mask')}
    reference = FreestreamReference(**read_json(args.reference))
    ans = Composite().forward(fields, reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **ans, x=fields['x'], y=fields['y'])
    write_json(args.output.with_suffix('.json'), dict(input_sha256=sha256(args.input),
        reference_sha256=sha256(args.reference), freeze_sha256=sha256(OUT/'FREEZE.json'),
        output_sha256=sha256(args.output), branch='ML_ONLY',
        accuracy_evaluated=False, untouched_status='requires separate exposure and source qualification'))
    verify_freeze()


if __name__ == '__main__': main()
