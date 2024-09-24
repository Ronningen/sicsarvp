# add "import cartopy" to the top of your jupyter notebook,
# before using these functions, or visualizations will fail

from tqdm.notebook import tqdm, trange
import global_land_mask as glm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from imageio import mimwrite
import numpy as np
import cv2


def fix_quiver_bug(field, lat):
    ufield, vfield = field
    old_magnitude = np.sqrt(ufield ** 2 + vfield ** 2)
    ufield_fixed = ufield / np.cos(np.radians(lat))
    new_magnitude = np.sqrt(ufield_fixed ** 2 + vfield ** 2)
    field_fixed = np.stack([ufield_fixed, vfield]) / new_magnitude * old_magnitude
    return field_fixed
    
#old: lim=True, clon = 73.2, 2.72*ncols
def create_cartopy(nrows=1, ncols=1, figsize=None, lim=False, clon=74):
    if not figsize: 
        figsize = (4*ncols, 8*nrows)
    fig, axs = plt.subplots(nrows, ncols,
        figsize=figsize,
        subplot_kw={
            'projection': ccrs.NorthPolarStereo(central_longitude=clon),
        }
    )
    if nrows>1 or ncols>1:
        for ax in np.array(axs).ravel():
            if lim:
                ax.set_extent([70.54,76,67.1,74], crs=ccrs.PlateCarree())
            ax.set_facecolor(cfeature.COLORS['water'])
            ax.add_feature(cfeature.LAND, edgecolor='black', zorder=0)
    else:
        if lim:
            axs.set_extent([70.54,76,67.1,74], crs=ccrs.PlateCarree())
        axs.set_facecolor(cfeature.COLORS['water'])
        axs.add_feature(cfeature.LAND, edgecolor='black', zorder=0)
    return fig, axs


def visualize_scalar_field(ax, lon, lat, field, colorbar=False, **kwargs):
    layer = ax.pcolormesh(
        lon,
        lat,
        field,
        transform=ccrs.PlateCarree(),
        **kwargs
    )
    if colorbar:
        plt.colorbar(layer, shrink=0.75)


def visualize_vector_field(ax, lon, lat, field, key_length=50, key_units='cm/s', key_color='black', **kwargs):
    field_fixed = fix_quiver_bug(field, lat)  # fix bug in quiver vector plot interpretation
    layer = ax.quiver(
        lon,
        lat,
        field_fixed[0],
        field_fixed[1],
        transform=ccrs.PlateCarree(),
        color=key_color,
        **kwargs
    )
    ax.quiverkey(layer, X=0.82, Y=0.2, U=key_length, label=f'{key_length} {key_units}',
                 labelpos='E', coordinates='figure')


def cartovideo(lon, lat, predictions, titles=None, fname='video', fps=12, **kwargs):
    mask = glm.is_land(lat, lon)
    frames = []
    for i in trange(len(predictions[0])):
        fig, axs = create_cartopy(1, len(predictions))
        if len(predictions)==1:
            axs = [axs]
        canvas = FigureCanvasAgg(fig)
        for ax, ps, t in zip(axs, predictions, titles):
            ax.set_title(t)
            frame = ps[i].copy()
            frame[(frame==0)] = np.nan
            frame[mask] = np.nan
            visualize_scalar_field(ax, lon, lat, frame, **kwargs)
        plt.suptitle(fname)
        canvas.draw()
        frames.append(np.array(canvas.buffer_rgba()))
        plt.close()
    mimwrite(fname+'.mp4', frames, fps=fps)

def video(lon, lat, predictions, titles=None, fname='video', fps=12, **kwargs):
    mask = glm.is_land(lat, lon)
    frames = []
    for i in trange(len(predictions[0])):
        fig, axs = plt.subplots(1, len(predictions), figsize=(8,8))
        if len(predictions)==1:
            axs = [axs]
        canvas = FigureCanvasAgg(fig)
        for ax, ps, t in zip(axs, predictions, titles):
            ax.set_title(t)
            frame = ps[i].copy()
            frame[(frame==0)] = np.nan
            frame[mask] = np.nan
            ax.imshow(frame, **kwargs)
        plt.suptitle(fname)
        canvas.draw()
        frames.append(np.array(canvas.buffer_rgba()))
        plt.close()
    mimwrite(fname+'.mp4', frames, fps=fps)

def fastvideo(predictions, fname='video', fps=12, vmin=0, vmax=1, **kwargs):
    arr_0 = np.ones((predictions[0].shape[0], predictions[0].shape[1], 3))*vmin
    arr_1 = np.ones((predictions[0].shape[0], predictions[0].shape[1], 3))*vmax
    split = np.concatenate([arr_0, arr_1, arr_0], 2)

    ars = [predictions[0]]
    for i in range(1, len(predictions)):
        ars.append(split)
        ars.append(predictions[i])
    gray_frames = np.clip((np.concatenate(ars, 2)-vmin)/(vmax-vmin), 0, 1)
    
    out = cv2.VideoWriter(fname+'.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (gray_frames.shape[2], gray_frames.shape[1]), False)
    for frame in tqdm(gray_frames, leave=False):
        out.write((frame*255).astype(np.uint8))
    out.release()