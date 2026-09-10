import h5py, numpy as np, glob, os, json
D='datasets/polishing/single_cam/20260901_1505/imitation_form'
paths=sorted(glob.glob(D+'/episode_*.hdf5'), key=lambda p:int(p.split('_')[-1].split('.')[0]))
rows=[]
frame_acc=None; frame_n=0
img_means={}
for p in paths:
    ei=int(p.split('_')[-1].split('.')[0])
    with h5py.File(p,'r') as f:
        op=np.asarray(f['observations/position'],np.float64)   # (T,6)
        of=np.asarray(f['observations/force'],np.float64)      # (T,3)
        ap=np.asarray(f['action/position'],np.float64)
        af=np.asarray(f['action/force'],np.float64)
        T=op.shape[0]
        imgs=np.asarray(f['observations/images/cam0'][::40]).astype(np.float32)  # (~19,H,W,3)
    fmag=np.linalg.norm(of,axis=1)
    fz=of[:,2]
    # contact detection: |F| above baseline+thresh
    base=np.median(fmag[:20])
    thr=max(2.0, base+2.0)
    contact = fmag > thr
    # count contact segments (runs of True), min length 5
    runs=[]; s=None
    for i,c in enumerate(np.r_[contact,False]):
        if c and s is None: s=i
        elif not c and s is not None:
            if i-s>=5: runs.append((s,i)); s=None
            else: s=None
    contact_frac=contact.mean()
    # z during contact vs non-contact
    z_contact = op[contact,2]
    row=dict(ep=ei, T=T,
        x0=op[0,0],y0=op[0,1],z0=op[0,2],
        rz0=op[0,5],
        xmin=op[:,0].min(),xmax=op[:,0].max(),
        ymin=op[:,1].min(),ymax=op[:,1].max(),
        zmin=op[:,2].min(),zmax=op[:,2].max(),
        z_contact_med=(np.median(z_contact) if z_contact.size else np.nan),
        z_contact_p10=(np.percentile(z_contact,10) if z_contact.size else np.nan),
        n_contact_seg=len(runs),
        contact_frac=contact_frac,
        f_base=base,
        fmag_p99=np.percentile(fmag,99), fmag_max=fmag.max(),
        fz_min=fz.min(), fz_mean_contact=(fz[contact].mean() if contact.any() else np.nan),
        of_start=np.linalg.norm(of[0]), of_end=np.linalg.norm(of[-1]),
        img_mean=float(imgs.mean()), img_std=float(imgs.std()),
        act_pos_dev=float(np.abs(ap-op).mean()),
    )
    rows.append(row)
    # accumulate a downsampled mean frame for early/late comparison
    gm=imgs.mean(axis=-1).mean(axis=0)  # (H,W)
    img_means[ei]=gm

keys=list(rows[0].keys())
print("ep  "+ " ".join(f"{k:>13}" for k in keys[1:]))
for r in rows:
    print(f"{r['ep']:3d} "+" ".join(f"{r[k]:13.3f}" if isinstance(r[k],float) else f"{r[k]:13d}" for k in keys[1:]))

np.save('/tmp/claude-1000/-home-eunseop-nrs-imitation/e70b8eeb-35ed-4fdf-8456-9c2d3df4c4f7/scratchpad/img_means_1505.npy',
        np.stack([img_means[i] for i in sorted(img_means)]))
json.dump(rows, open('/tmp/claude-1000/-home-eunseop-nrs-imitation/e70b8eeb-35ed-4fdf-8456-9c2d3df4c4f7/scratchpad/rows_1505.json','w'), default=float, indent=0)
print("\nsaved img_means_1505.npy, rows_1505.json")
