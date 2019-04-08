# pyLucid

A python implementation of Lucid Data Dreaming.

Lucid Data Dreaming is a data augmentation technique for semi-supervised video object segmentation, which is proposed
in [Lucid Data Dreaming for Multiple Object Tracking](https://arxiv.org/abs/1703.09554), A. Khoreva, R. Benenson, E. Ilg, T. Brox and B. Schiele, arXiv preprint arXiv:1703.09554, 2017.

This implementation is based on the offcial released code written in matlab, which can be found [here](https://github.com/ankhoreva/LucidDataDreaming).
I also used code from harveyslash's PatchMatch repository, his work can be found [here](https://github.com/harveyslash/PatchMatch).

## Dependencies

- opencv (3.2.0)
- numpy
- pycuda

## Usage

To generate a pair of images, you can refer to `demo.py`.
We firstly generate the background image and then used it to
do the Lucid Data Dreaming. The former invokes `patchPaint.py` and can be done in about one minute. Once the background is generated,
there is no need to do the same work again. The Lucid Data Dreaming
invokes `lucidDream.py`, using only around 0.4 seconds to generate a pair of images, which is much faster than the matlab version.
> (The time mentioned above is on a server with NVIDIA TITAN X GPU and Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz)
