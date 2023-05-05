import numpy as np
from skimage.feature import canny
from PIL import Image
from skimage.morphology import dilation
from skimage.measure import label
from skimage.measure import regionprops


def do_bboxes_overlap(a, b):
    return (
        a[0] < b[2] and
        a[2] > b[0] and
        a[1] < b[3] and
        a[3] > b[1]
    )

def merge_bboxes(a, b):
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3])
    )

def are_bboxes_aligned(a, b, axis):
    return (
        a[0 + axis] < b[2 + axis] and
        b[0 + axis] < a[2 + axis]
    )

def cluster_bboxes(bboxes, axis=0):

    clusters = []

    # Regroup bboxes which overlap along the current axis.
    # For instance, two panels on the same row overlap
    # along their verticial coordinate.
    for bbox in bboxes:
        for cluster in clusters:
            if any(
                are_bboxes_aligned(b, bbox, axis=axis)
                for b in cluster
            ):
                cluster.append(bbox)
                break
        else:
            clusters.append([bbox])

    # We want rows to be ordered from top to bottom, and
    # columns to be ordered from left to right.
    clusters.sort(key=lambda c: c[0][0 + axis])

    # For each row, we want to cluster the panels of that
    # row into columns, etc. etc.
    for i, cluster in enumerate(clusters):
        if len(cluster) > 1:
            clusters[i] = cluster_bboxes(
                bboxes=cluster,
                axis=1 if axis == 0 else 0
            )

    return clusters

def flatten(l):
    for el in l:
        if isinstance(el, list):
            yield from flatten(el)
        else:
            yield el

def finder(path, imagename):
    file = str(path/imagename)
    if not imagename[-4::1]==".jpg":
        return;
    im = np.array(Image.open(file).convert('L'))
    
    edges = canny(im)
    
    dilate = dilation(dilation(dilation(dilation(edges))))
    
    labels = label(dilate)
    
    regions = regionprops(labels)
    panels = []
    for region in regions[1:]:

        for i, panel in enumerate(panels):
            if do_bboxes_overlap(region.bbox, panel):
                panels[i] = merge_bboxes(panel, region.bbox)
                break
        else:
            panels.append(region.bbox)
    
    for i, bbox in reversed(list(enumerate(panels))):
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < 0.01 * im.shape[0] * im.shape[1]:
            del panels[i]
    
    panel_img = np.zeros_like(labels)

    for i, bbox in enumerate(panels, start=1):
        panel_img[bbox[0]:bbox[2], bbox[1]:bbox[3]] = i

    clusters = cluster_bboxes(panels)
    
    return clusters