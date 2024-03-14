import os
import nrrd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from skimage.transform import rescale
from allensdk.api.queries.ontologies_api import OntologiesApi
from allensdk.core.structure_tree import StructureTree
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
from allensdk.config.manifest import Manifest
from allensdk.core.reference_space import ReferenceSpace


def rotate_nn(data, angle, axes):
    """
    Rotate a `data` based on rotating coordinates.
    """

    # Create grid of indices
    shape = data.shape
    d1, d2, d3 = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]

    # Rotate the indices
    d1r = rotate(d1, angle=angle, axes=axes)
    d2r = rotate(d2, angle=angle, axes=axes)
    d3r = rotate(d3, angle=angle, axes=axes)

    # Round to integer indices
    d1r = np.round(d1r)
    d2r = np.round(d2r)
    d3r = np.round(d3r)

    d1r = np.clip(d1r, 0, shape[0])
    d2r = np.clip(d2r, 0, shape[1])
    d3r = np.clip(d3r, 0, shape[2])

    return data[d1r, d2r, d3r]


def project_top_2D(data):
    from itertools import product
    flat_top = np.zeros_like(data[:, 0, :])
    for x, y in product(list(range(data.shape[0])), list(range(data.shape[2]))):
        first_val_loc = np.where(data[x, :, y] != 0)[0]

        if len(first_val_loc) > 0:
            flat_top[x, y] = data[x, first_val_loc[0], y]
        else:
            continue

    return flat_top

if __name__ == '__main__':
    output_path = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Pol_Bech\Parameters\Widefield\allen_brain"
    oapi = OntologiesApi()
    structure_graph = oapi.get_structures_with_sets([1])  # 1 is the id of the adult mouse structure graph

    # This removes some unused fields returned by the query
    structure_graph = StructureTree.clean_structures(structure_graph)

    tree = StructureTree(structure_graph)

    # the annotation download writes a file, so we will need somwhere to put it
    annotation_dir = '../nwb_utils/annotation'
    Manifest.safe_mkdir(annotation_dir)

    annotation_path = os.path.join(annotation_dir, 'annotation.nrrd')

    # this is a string which contains the name of the latest ccf version
    annotation_version = MouseConnectivityApi.CCF_VERSION_DEFAULT

    mcapi = MouseConnectivityApi()
    mcapi.download_annotation_volume(annotation_version, 25, annotation_path) # 25 micron ccf

    annotation, meta = nrrd.read(annotation_path)
    rsp = ReferenceSpace(tree, annotation, [25, 25, 25])

    bregma_origin = (216, 18, 228) # according to https://community.brain-map.org/t/how-to-transform-ccf-x-y-z-coordinates-into-stereotactic-coordinates/1858
    bregma_volume = np.zeros_like(rsp.annotation)
    bregma_volume[bregma_origin] = 1

    angle_ap = -30
    angle_ml = 5
    axes_ap = (1, 2) # AP, DV, ML, rotate along AP axis
    axes_ml = (0, 1)
    data_r = rotate_nn(rsp.annotation, angle_ml, axes_ml)
    data_r = rotate_nn(data_r, angle_ap, axes_ap)

    iso = rotate_nn(rsp.make_structure_mask([315]), angle_ml, axes_ml)
    iso = rotate_nn(iso, angle_ap, axes_ap)

    flat_brain = project_top_2D(data_r)
    iso_mask = project_top_2D(iso)

    bregma_rot = rotate_nn(bregma_volume, angle_ml, axes_ml)
    bregma_rot = rotate_nn(bregma_volume, angle_ap, axes_ap)

    bregma_coords_new = np.where(bregma_rot != 0)
    bregma_coords_flat = (bregma_coords_new[0], bregma_coords_new[2])

    fig, ax =plt.subplots()
    plt.axis('scaled')
    ax.contour(flat_brain, levels=np.unique(flat_brain), colors='gray')
    ax.scatter(bregma_coords_flat[1], bregma_coords_flat[0], marker='+', c='k', s=100, linewidths=2)
    ax.set_xlim([0, flat_brain.shape[1]])
    ax.set_ylim([0, flat_brain.shape[0]])
    ax.invert_yaxis()
    fig.show()
    fig.savefig(output_path + r"\allen_contours_30deg.svg")
    fig.savefig(output_path + r"\allen_contours_30deg.png")

    structure_list = ["FRP", "MOp", "MOs", "SSp-n", "SSp-bfd", "SSp-ll", "SSp-m", "SSp-ul", "SSp-tr", "SSp-un", "SSs",
                      "GU", "VISal", "VISam", "VISal", "VISp", "VISpl", "VISpm", "VISli", "VISpor", "ORB", "PL", "ILA", "AI", "RSP", "PTLp",
                      "TEa", "PERI", "ECT"]

    x, y = np.meshgrid(np.linspace(0, flat_brain.shape[1], flat_brain.shape[1], dtype=int),
                       np.linspace(0, flat_brain.shape[0], flat_brain.shape[0], dtype=int))
    fig, ax = plt.subplots()
    plt.axis('scaled')
    ax.contour(iso_mask,colors='k')
    ax.scatter(bregma_coords_flat[1], bregma_coords_flat[0], marker='+', c='k', s=100, linewidths=2, zorder=2)
    ax.set_xlim([90, 370])
    ax.set_ylim([80, 440])
    ax.invert_yaxis()
    fig.show()
    fig.savefig(output_path + r"\allen_contours_30deg_isocortex.svg")
    fig.savefig(output_path + r"\allen_contours_30deg_isocortex.png")

    fig, ax = plt.subplots()
    plt.axis('scaled')
    ax.contour(np.where(iso_mask>0, flat_brain,0), levels=np.unique(np.where(iso_mask>0, flat_brain,0)), colors='gray')
    ax.contour(iso_mask,colors='k')
    ax.scatter(bregma_coords_flat[1], bregma_coords_flat[0], marker='+', c='k', s=100, linewidths=2, zorder=2)
    ax.set_xlim([90, 370])
    ax.set_ylim([80, 440])
    ax.invert_yaxis()
    fig.show()
    fig.savefig(output_path + r"\allen_contours_30deg_areas.svg")
    fig.savefig(output_path + r"\allen_contours_30deg_areas.png")

    bregma_coords_zoom = (216 - 80, 348 - 80)
    zoom_brain = np.where(iso_mask > 0, flat_brain, np.nan)[80:440, 80:360]
    zoom_iso = iso_mask[80:440, 80:360]
    fig, ax = plt.subplots()
    plt.axis('scaled')
    ax.contour(zoom_brain, levels=np.unique(zoom_brain), colors='gray')
    ax.contour(zoom_iso, colors='k')
    ax.scatter(bregma_coords_zoom[1], bregma_coords_zoom[0], marker='+', c='k', s=100, linewidths=2, zorder=2)
    ax.set_xlim([0, zoom_brain.shape[1]])
    ax.set_ylim([0, zoom_brain.shape[0]])
    ax.invert_yaxis()
    fig.show()
    fig.savefig(output_path + r"\allen_contours_30deg_zoom.svg")
    fig.savefig(output_path + r"\allen_contours_30deg_zoom.png")

    fig, ax = plt.subplots()
    plt.axis('scaled')
    ax.contour(np.round(rescale(zoom_brain, 1.8, anti_aliasing=False)), levels=np.unique(zoom_brain), colors='gray',
               linewidths=1)
    ax.contour(rescale(zoom_iso, 1.8, anti_aliasing=False), colors='k', linewidths=1, zorder=1)
    ax.scatter(bregma_coords_zoom[1] * 1.8, bregma_coords_zoom[0] * 1.8, marker='+', c='k', s=100, linewidths=2,
               zorder=2)
    ax.set_xlim([0, 500])
    ax.set_ylim([0, 640])
    ax.invert_yaxis()
    fig.show()
    fig.savefig(output_path + r"\allen_contours_30deg_500x640.svg")
    fig.savefig(output_path + r"\allen_contours_30deg_500x640.png")

    np.save(output_path+r"\allen_brain_tilted_500x640.npy", np.round(rescale(zoom_brain, 1.8, anti_aliasing=False)))
    np.save(output_path+r"\allen_isocortex_tilted_500x640.npy", rescale(zoom_iso, 1.8, anti_aliasing=False))
    np.save(output_path+r"\allen_bregma_tilted_500x640.npy", (bregma_coords_zoom[1]*1.8, bregma_coords_zoom[0]*1.8))