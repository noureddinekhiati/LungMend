"""NIfTI reading / writing in the layout the model was trained on.

The model works on (z, y, x) arrays: apex first, anterior first, at about 0.6 mm isotropic with the voxel size taken as 1.
- layout "ras": any correctly oriented NIfTI (e.g. dcm2niix output). The volume is reoriented and resampled to 0.6 mm, and the
  result is mapped back onto the input grid.
- layout "zyx": arrays already stored as (z, y, x) on a ~0.6 mm grid (the files of the paper's evaluation). Used as is.
"""
import nibabel as nib
import numpy as np

from .preprocess import resize

ISO_MM = 0.6


class Volume:
    def __init__(self, path, layout="ras", order=0):
        self.img = nib.load(str(path))
        self.layout = layout
        if layout == "zyx":
            self.data = np.asanyarray(self.img.dataobj)
            return
        self.canon = nib.as_closest_canonical(self.img)
        xyz = np.asanyarray(self.canon.dataobj)
        zyx = xyz.transpose(2, 1, 0)[::-1, ::-1]  # superior -> inferior, anterior -> posterior
        self.zyx_shape = zyx.shape
        sp = np.array(self.canon.header.get_zooms()[:3])[::-1]
        iso_shape = np.maximum(np.round(np.array(zyx.shape) * sp / ISO_MM).astype(int), 1)
        self.data = resize(zyx, iso_shape, order=order)

    def save_like(self, arr, path):
        """Write a (z, y, x) array computed on self.data's grid onto the grid of the input file."""
        arr = arr.astype(np.uint8)
        if self.layout == "zyx":
            nib.save(nib.Nifti1Image(arr, self.img.affine, self.img.header), str(path))
            return
        zyx = (resize(arr, self.zyx_shape, order=0) > 0.5).astype(np.uint8)
        xyz = np.ascontiguousarray(zyx[::-1, ::-1].transpose(2, 1, 0))
        out = nib.Nifti1Image(xyz, self.canon.affine)
        ornt = nib.orientations.ornt_transform(nib.io_orientation(self.canon.affine), nib.io_orientation(self.img.affine))
        out = out.as_reoriented(ornt)
        hdr = self.img.header.copy()
        hdr.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(np.asanyarray(out.dataobj), self.img.affine, hdr), str(path))
