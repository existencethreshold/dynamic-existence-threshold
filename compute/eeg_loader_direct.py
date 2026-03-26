"""Direct EEG loader - scans filesystem instead of using mne.datasets.sleep_physionet"""
import os
import mne

STAGE_MAP = {
    'Sleep stage W': 'W', 'Sleep stage 1': 'N1', 'Sleep stage 2': 'N2',
    'Sleep stage 3': 'N3', 'Sleep stage 4': 'N3', 'Sleep stage R': 'R',
    'Sleep stage ?': '?', 'Movement time': 'MT',
}

def load_dataset(n_subjects=50):
    DATA_DIR = os.path.join(os.environ.get("SLEEP_EDF_DIR", "./external/sleep-edf"), "physionet-sleep-data")
    psg_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('-PSG.edf') and not f.startswith('tmp')])

    pairs = []
    all_files = os.listdir(DATA_DIR)
    for psg in psg_files:
        prefix = psg.split('E0-PSG')[0]
        # Hypnogram suffix varies per subject (EC, EH, EJ, EP, EV, EU, EA, etc.)
        hyps = [f for f in all_files if f.startswith(prefix) and 'Hypnogram' in f
                and not f.startswith('tmp') and os.path.getsize(os.path.join(DATA_DIR, f)) > 100]
        if hyps:
            pairs.append((os.path.join(DATA_DIR, psg), os.path.join(DATA_DIR, hyps[0])))

    print(f"Found {len(pairs)} subject pairs, loading up to {n_subjects}...", flush=True)

    all_epochs = []
    loaded = 0
    for idx, (psg_path, hypno_path) in enumerate(pairs):
        if loaded >= n_subjects:
            break
        if os.path.getsize(psg_path) < 1000 or os.path.getsize(hypno_path) < 100:
            print(f"  Skipping {os.path.basename(psg_path)} (corrupt/empty)", flush=True)
            continue
        if loaded % 10 == 0:
            print(f"  Loading subject {loaded+1}/{n_subjects} ({os.path.basename(psg_path)})...", flush=True)
        try:
            raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
            eeg_channel = 'EEG Fpz-Cz'
            if eeg_channel not in raw.ch_names:
                eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch]
                eeg_channel = eeg_channels[0] if eeg_channels else raw.ch_names[0]
            raw.pick([eeg_channel])
            sfreq = raw.info['sfreq']
            raw_data = raw.get_data()[0]
            annot = mne.read_annotations(hypno_path)
            epoch_duration = 30.0
            epoch_samples = int(epoch_duration * sfreq)
            for onset, duration, description in zip(annot.onset, annot.duration, annot.description):
                stage = STAGE_MAP.get(description)
                if stage is None or stage in ('?', 'MT'):
                    continue
                n_ep = int(duration / epoch_duration)
                start_sample = int(onset * sfreq)
                for i in range(n_ep):
                    ep_start = start_sample + i * epoch_samples
                    ep_end = ep_start + epoch_samples
                    if ep_end > len(raw_data):
                        break
                    all_epochs.append({'data': raw_data[ep_start:ep_end], 'stage': stage, 'sfreq': sfreq})
            loaded += 1
        except Exception as e:
            print(f"  Skipping {os.path.basename(psg_path)}: {e}", flush=True)
            continue

    stage_counts = {}
    for ep in all_epochs:
        stage_counts[ep['stage']] = stage_counts.get(ep['stage'], 0) + 1
    print(f"Loaded {len(all_epochs)} epochs from {loaded} subjects", flush=True)
    print(f"Stages: {stage_counts}", flush=True)
    return all_epochs
