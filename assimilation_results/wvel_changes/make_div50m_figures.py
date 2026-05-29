"""
Remake wvel_uvel_div50m_sw_difference.png and uvel_div50m_sw_difference.png
with divergence (and U, V maps) averaged over 0–50 m instead of 0–75 m.
Overwrites the existing files in place.
"""
import sys, warnings
sys.path.insert(0, '/home/edavenport/analysis/vel-assim-manuscript')
warnings.filterwarnings('ignore')

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean.cm as cmo
import xgcm
from xmitgcm import open_mdsdataset
from open_tpose import tpose2012to2013
from mitgcm_assim.ctrls import load_controls_and_sensitivities

# ── Config ────────────────────────────────────────────────────────────────────
grid_dir  = '/data/SO6/TPOSE_diags/tpose6/grid_6/'
itPerFile = 72
offset    = 10
lon_slice = slice(140, 275)
lat_slice = slice(-17, 17)
moorings  = [(0, 190), (0, 220), (0, 250)]

# ── Load TPOSE-Vel (state only) ───────────────────────────────────────────────
print('Loading TPOSE-Vel...')
data_dir = '/data/SO3/edavenport/tpose6/sep2012/run_iter14/'
num_diags = 30 + 31 + offset
intervals = range(itPerFile, itPerFile * num_diags, itPerFile)
ds_vel = open_mdsdataset(data_dir=data_dir, grid_dir=grid_dir, iters=intervals,
                         prefix=['diag_state'], ref_date='2012-09-01', delta_t=1200)
for data_dir2, num_d, ref in [
    ('/data/SO3/edavenport/tpose6/nov2012/run_iter20/', 30+31+offset, '2012-11-01'),
    ('/data/SO3/edavenport/tpose6/jan2013/run_iter14/', 31+28+offset, '2013-01-01'),
    ('/data/SO3/edavenport/tpose6/mar2013/run_iter16/', 31+30+31+30, '2013-03-01'),
]:
    ivals = range(itPerFile * offset, itPerFile * num_d, itPerFile)
    ds_vel = xr.concat([ds_vel,
        open_mdsdataset(data_dir=data_dir2, grid_dir=grid_dir, iters=ivals,
                        prefix=['diag_state'], ref_date=ref, delta_t=1200)], dim='time')
for c in ('XC','YC','Z','Zl','XG','YG'):
    if c in ds_vel.coords: ds_vel[c] = ds_vel[c].astype(float)
print(f'  Vel: {dict(ds_vel.dims)}')

# ── Load TPOSE-noVel ─────────────────────────────────────────────────────────
print('Loading TPOSE-noVel...')
ds = tpose2012to2013(['diag_state'])
for c in ('XC','YC','Z','Zl','XG','YG'):
    if c in ds.coords: ds[c] = ds[c].astype(float)
ds = ds.sel(time=slice('2012-09-01','2013-06-30'))
print(f'  noVel: {dict(ds.dims)}')

# ── Divergence (0–50 m) ───────────────────────────────────────────────────────
def compute_div_30m(ds_run):
    """Horizontal divergence averaged 0–50 m, time-mean (s⁻¹)."""
    grid_r = xgcm.Grid(ds_run, periodic=['X','Y'])
    u_30m = ds_run.UVEL.where(ds_run.Z >= -30).mean(['Z','time']).compute()
    v_30m = ds_run.VVEL.where(ds_run.Z >= -30).mean(['Z','time']).compute()
    div = (grid_r.diff(u_30m, 'X', boundary='extend') / ds_run.dxF +
           grid_r.diff(v_30m, 'Y', boundary='extend') / ds_run.dyF)
    return div.compute()

print('Computing divergence noVel...')
div_noVel = compute_div_30m(ds)
print('Computing divergence Vel...')
div_vel   = compute_div_30m(ds_vel)
div_diff  = div_vel - div_noVel
print(f'  noVel p1/p99: {float(div_noVel.quantile(0.01)):.2e} / {float(div_noVel.quantile(0.99)):.2e}')
print(f'  Vel   p1/p99: {float(div_vel.quantile(0.01)):.2e} / {float(div_vel.quantile(0.99)):.2e}')

# ── Shortwave controls ────────────────────────────────────────────────────────
print('Loading SW controls...')
ctrl_vel_segs, ctrl_noVel_segs = [], []
for data_dir2, fi in [('/data/SO3/edavenport/tpose6/sep2012/run_iter14/',14),
                       ('/data/SO3/edavenport/tpose6/nov2012/run_iter20/',20),
                       ('/data/SO3/edavenport/tpose6/jan2013/run_iter14/',14),
                       ('/data/SO3/edavenport/tpose6/mar2013/run_iter16/',16)]:
    dc = load_controls_and_sensitivities(data_dir2, grid_dir, fi)
    ctrl_vel_segs.append(dc.xx_swdown)
for data_dir2, fi in [('/data/SO6/TPOSE/tpose6/sep2012/ITER7/',7),
                       ('/data/SO6/TPOSE/tpose6/nov2012/ITER4/',4),
                       ('/data/SO6/TPOSE/tpose6/jan2013/ITER7/',7),
                       ('/data/SO6/TPOSE/tpose6/mar2013/ITER4/',4)]:
    dc = load_controls_and_sensitivities(data_dir2, grid_dir, fi)
    ctrl_noVel_segs.append(dc.xx_swdown)

sw_vel   = xr.concat(ctrl_vel_segs,   dim='time').mean('time').compute()
sw_noVel = xr.concat(ctrl_noVel_segs, dim='time').mean('time').compute()
for _da in [sw_vel, sw_noVel]:
    for _c in ('XC','YC'):
        if _c in _da.coords: _da[_c] = _da[_c].values.astype(float)
sw_diff = sw_vel - sw_noVel
for _c in ('XC','YC'):
    if _c in sw_diff.coords: sw_diff[_c] = sw_diff[_c].values.astype(float)

# ── Helper: levels ────────────────────────────────────────────────────────────
def sym_levels(arrs, pct=99, n=51):
    v = float(np.nanpercentile(np.abs(np.concatenate([a.values.ravel() for a in arrs])), pct))
    return np.linspace(-v, v, n)

# ── 5-column figure ───────────────────────────────────────────────────────────
def make_5col(outfile):
    fig, ax = plt.subplots(figsize=(32,9), nrows=3, ncols=5, sharex=True)

    levels_u_sec = np.linspace(-1.0, 1.0, 150)
    lev_u30  = sym_levels([
        ds.UVEL.sel(YC=lat_slice,XG=lon_slice,Z=slice(0,-50)).mean(['time','Z']),
        ds_vel.UVEL.sel(YC=lat_slice,XG=lon_slice,Z=slice(0,-50)).mean(['time','Z'])])
    lev_v30  = sym_levels([
        ds.VVEL.sel(YG=lat_slice,XC=lon_slice,Z=slice(0,-50)).mean(['time','Z']),
        ds_vel.VVEL.sel(YG=lat_slice,XC=lon_slice,Z=slice(0,-50)).mean(['time','Z'])])
    lev_div  = sym_levels([div_noVel.sel(XC=lon_slice,YC=lat_slice),
                            div_vel.sel(XC=lon_slice,YC=lat_slice)])
    lev_sw   = sym_levels([sw_noVel.sel(XC=lon_slice,YC=lat_slice),
                            sw_vel.sel(XC=lon_slice,YC=lat_slice)])

    rows_def = [('TPOSE-noVel', ds, div_noVel, sw_noVel),
                ('TPOSE-Vel',   ds_vel, div_vel, sw_vel),
                ('Difference',  None,   div_diff, sw_diff)]

    for row, (label, dset, div_da, sw_da) in enumerate(rows_def):
        # Col 0: UVEL section at 0N
        if label == 'Difference':
            d0 = (ds_vel.UVEL.sel(YC=0,method='nearest').sel(XG=lon_slice).mean('time') -
                  ds.UVEL.sel(YC=0,method='nearest').sel(XG=lon_slice).mean('time'))
        else:
            d0 = dset.UVEL.sel(YC=0,method='nearest').sel(XG=lon_slice).mean('time')
        d0.plot.contourf(ax=ax[row,0], levels=levels_u_sec, cmap=cmo.balance,
                         cbar_kwargs={'label':'(m/s)'})
        for xv in [190,220,250]: ax[row,0].axvline(xv,color='k',linestyle='--')
        ax[row,0].set_ylim(-300,0)
        ax[row,0].set_title(f'{label}, avg U at 0°N')
        ax[row,0].set_xlabel('' if row<2 else 'Longitude')
        ax[row,0].set_ylabel('Depth (m)')

        # Col 1: UVEL map 0–50 m
        if label == 'Difference':
            d1 = (ds_vel.UVEL.sel(YC=lat_slice,XG=lon_slice,Z=slice(0,-50)).mean(['time','Z']) -
                  ds.UVEL.sel(YC=lat_slice,XG=lon_slice,Z=slice(0,-50)).mean(['time','Z']))
            lev1 = sym_levels([d1])
        else:
            d1 = dset.UVEL.sel(YC=lat_slice,XG=lon_slice,Z=slice(0,-50)).mean(['time','Z'])
            lev1 = lev_u30
        d1.plot.contourf(ax=ax[row,1], levels=lev1, cmap=cmo.balance,
                         cbar_kwargs={'label':'(m/s)'})
        for lat,lon in moorings: ax[row,1].scatter(lon,lat,color='k',marker='x',s=75)
        ax[row,1].set_title(f'{label}, avg U 0–50 m')
        ax[row,1].set_xlabel('' if row<2 else 'Longitude')
        ax[row,1].set_ylabel('Latitude')

        # Col 2: VVEL map 0–50 m
        if label == 'Difference':
            d2 = (ds_vel.VVEL.sel(YG=lat_slice,XC=lon_slice,Z=slice(0,-50)).mean(['time','Z']) -
                  ds.VVEL.sel(YG=lat_slice,XC=lon_slice,Z=slice(0,-50)).mean(['time','Z']))
            lev2 = sym_levels([d2])
        else:
            d2 = dset.VVEL.sel(YG=lat_slice,XC=lon_slice,Z=slice(0,-50)).mean(['time','Z'])
            lev2 = lev_v30
        d2.plot.contourf(ax=ax[row,2], levels=lev2, cmap=cmo.balance,
                         cbar_kwargs={'label':'(m/s)'})
        for lat,lon in moorings: ax[row,2].scatter(lon,lat,color='k',marker='x',s=75)
        ax[row,2].set_title(f'{label}, avg V 0–50 m')
        ax[row,2].set_xlabel('' if row<2 else 'Longitude')
        ax[row,2].set_ylabel('Latitude')

        # Col 3: Divergence 0–50 m
        lev3 = sym_levels([div_diff]) if label=='Difference' else lev_div
        div_da.sel(XC=lon_slice,YC=lat_slice).plot.contourf(
            ax=ax[row,3], levels=lev3, cmap=cmo.balance, cbar_kwargs={'label':'(s⁻¹)'})
        for lat,lon in moorings: ax[row,3].scatter(lon,lat,color='k',marker='x',s=75)
        ax[row,3].set_title(f'{label}, ∂u/∂x+∂v/∂y, 0–50 m')
        ax[row,3].set_xlabel('' if row<2 else 'Longitude')
        ax[row,3].set_ylabel('Latitude')

        # Col 4: SW control
        lev4 = sym_levels([sw_diff]) if label=='Difference' else lev_sw
        sw_da.sel(XC=lon_slice,YC=lat_slice).plot.contourf(
            ax=ax[row,4], levels=lev4, cmap=cmo.balance,
            cbar_kwargs={'label':'(units of σ)'})
        for lat,lon in moorings: ax[row,4].scatter(lon,lat,color='k',marker='x',s=75)
        ax[row,4].set_title(f'{label}, SW ctrl adj. (mean)')
        ax[row,4].set_xlabel('' if row<2 else 'Longitude')
        ax[row,4].set_ylabel('Latitude')

    plt.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved {outfile}')

# ── 3-column figure ───────────────────────────────────────────────────────────
def make_3col(outfile):
    fig, ax = plt.subplots(figsize=(20,9), nrows=3, ncols=3, sharex=True)

    levels_u_sec = np.linspace(-1.0, 1.0, 150)
    lev_div  = sym_levels([div_noVel.sel(XC=lon_slice,YC=lat_slice),
                            div_vel.sel(XC=lon_slice,YC=lat_slice)])
    lev_sw   = sym_levels([sw_noVel.sel(XC=lon_slice,YC=lat_slice),
                            sw_vel.sel(XC=lon_slice,YC=lat_slice)])

    rows_def = [('TPOSE-noVel', ds, div_noVel, sw_noVel),
                ('TPOSE-Vel',   ds_vel, div_vel, sw_vel),
                ('Difference',  None,   div_diff, sw_diff)]

    for row, (label, dset, div_da, sw_da) in enumerate(rows_def):
        # Col 0: UVEL section at 0N
        if label == 'Difference':
            d0 = (ds_vel.UVEL.sel(YC=0,method='nearest').sel(XG=lon_slice).mean('time') -
                  ds.UVEL.sel(YC=0,method='nearest').sel(XG=lon_slice).mean('time'))
        else:
            d0 = dset.UVEL.sel(YC=0,method='nearest').sel(XG=lon_slice).mean('time')
        d0.plot.contourf(ax=ax[row,0], levels=levels_u_sec, cmap=cmo.balance,
                         cbar_kwargs={'label':'(m/s)'})
        for xv in [190,220,250]: ax[row,0].axvline(xv,color='k',linestyle='--')
        ax[row,0].set_ylim(-300,0)
        ax[row,0].set_title(f'{label}, avg U at 0°N')
        ax[row,0].set_xlabel('' if row<2 else 'Longitude')
        ax[row,0].set_ylabel('Depth (m)')

        # Col 1: Divergence 0–50 m
        lev1 = sym_levels([div_diff]) if label=='Difference' else lev_div
        div_da.sel(XC=lon_slice,YC=lat_slice).plot.contourf(
            ax=ax[row,1], levels=lev1, cmap=cmo.balance, cbar_kwargs={'label':'(s⁻¹)'})
        for lat,lon in moorings: ax[row,1].scatter(lon,lat,color='k',marker='x',s=75)
        ax[row,1].set_title(f'{label}, ∂u/∂x+∂v/∂y, 0–50 m')
        ax[row,1].set_xlabel('' if row<2 else 'Longitude')
        ax[row,1].set_ylabel('Latitude')

        # Col 2: SW control
        lev2 = sym_levels([sw_diff]) if label=='Difference' else lev_sw
        sw_da.sel(XC=lon_slice,YC=lat_slice).plot.contourf(
            ax=ax[row,2], levels=lev2, cmap=cmo.balance,
            cbar_kwargs={'label':'(units of σ)'})
        for lat,lon in moorings: ax[row,2].scatter(lon,lat,color='k',marker='x',s=75)
        ax[row,2].set_title(f'{label}, SW ctrl adj. (mean)')
        ax[row,2].set_xlabel('' if row<2 else 'Longitude')
        ax[row,2].set_ylabel('Latitude')

    plt.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved {outfile}')

# ── Run ───────────────────────────────────────────────────────────────────────
make_5col('wvel_uvel_div50m_sw_difference.png')
make_3col('uvel_div50m_sw_difference.png')
