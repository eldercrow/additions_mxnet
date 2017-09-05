import numpy as np
from bbox_transform import bbox_transform


def transform_head(rois, head_bbs, grid_hw):
    '''
    Transform head bbs w.r.t rois.
    '''
    n_bbs = len(head_bbs)

    # normalize head_bbs
    head_bbs[:, 0::2] -= rois[:, 0:1]
    head_bbs[:, 0::2] /= rois[:, 2:3] - rois[:, 0:1]
    head_bbs[:, 1::2] -= rois[:, 1:2]
    head_bbs[:, 1::2] /= rois[:, 3:4] - rois[:, 1:2]

    # grid info
    ctr_grid, rect_grid = _get_grid_info(grid_hw)

    # bbox center
    ctr_bbs = (head_bbs[:, 2:] + head_bbs[:, 0:2]) / 2.0

    # compute grid idx
    dx = ctr_bbs[:, 0:1] - np.tile(np.transpose(ctr_grid[:, 0:1]), (n_bbs, 1))
    dy = ctr_bbs[:, 1:2] - np.tile(np.transpose(ctr_grid[:, 1:2]), (n_bbs, 1))
    dc = dx*dx + dy*dy
    gid = np.argmin(dc, axis=1)

    # compute transform target
    gt_bbs = rect_grid[gid, :]
    head_target = bbox_transform(head_bbs, gt_bbs)
    head_weight = np.ones_like(head_target)

    return np.reshape(gid, (-1, 1)), head_target, head_weight


def transform_joint(rois, joints, grid_hw):
    '''
    Transform joint w.r.t rois.
    '''
    n_joint = len(joints)

    # normalize joints
    joints[:, 0] -= rois[:, 0]
    joints[:, 1] -= rois[:, 1]

    # grid info
    ctr_grid, _ = _get_grid_info(grid_hw)

    # compute grid idx
    dx = joints[:, 0:1] - np.tile(np.transpose(ctr_grid[:, 0:1]), (n_bbs, 1))
    dy = joints[:, 1:2] - np.tile(np.transpose(ctr_grid[:, 1:2]), (n_bbs, 1))
    dc = dx*dx + dy*dy
    gid = np.argmin(dc, axis=1)

    # compute transform target
    joint_target = np.hstack((dx * gw, dy * gh))
    joint_weight = np.tile(joints[:, 2:3], (1, 2))

    return np.reshape(gid, (-1, 1)), joint_target, joint_weight


def _get_grid_info(grid_hw):
    ''' compute grid info '''
    gh, gw = grid_hw
    px = np.arange(gw+1) / float(gw)
    py = np.arange(gh+1) / float(gh)

    cx_grid = (px[1:] + px[:-1]) / 2.0
    cy_grid = (py[1:] + py[:-1]) / 2.0
    ctr_grid = _meshgrid_flatten(cx_grid, cy_grid) # (grid_h*grid_w, 2)

    xy0_grid = _meshgrid_flatten(px[:-1], py[:-1])
    xy1_grid = _meshgrid_flatten(px[1:], py[1:])
    rect_grid = np.hstack((xy0_grid, xy1_grid))

    return ctr_grid, rect_grid


def _meshgrid_flatten(px, py):
    gx, gy = np.meshgrid(px, py)
    gx = np.reshape(gx, (-1, 1))
    gy = np.reshape(gy, (-1, 1))

    return np.hstack((gx, gy))


# def bbox_transform(ex_rois, gt_rois):
#     """
#     compute bounding box regression targets from ex_rois to gt_rois
#     :param ex_rois: [N, 4]
#     :param gt_rois: [N, 4]
#     :return: [N, 4]
#     """
#     assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'
#
#     ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
#     ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
#     ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1.0)
#     ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1.0)
#
#     gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
#     gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
#     gt_ctr_x = gt_rois[:, 0] + 0.5 * (gt_widths - 1.0)
#     gt_ctr_y = gt_rois[:, 1] + 0.5 * (gt_heights - 1.0)
#
#     targets_dx = (gt_ctr_x - ex_ctr_x) / (ex_widths + 1e-14)
#     targets_dy = (gt_ctr_y - ex_ctr_y) / (ex_heights + 1e-14)
#     targets_dw = np.log(gt_widths / ex_widths)
#     targets_dh = np.log(gt_heights / ex_heights)
#
#     targets = np.vstack(
#         (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
#     return targets
#
# # unit test
# if __name__ == '__main__':
#     head_bbs = [[585, 131, 671, 234], [510, 93, 571, 167]]
#     head_bbs = np.array(head_bbs, dtype=np.float32)
#     print head_bbs
#
#     rois = [[540, 131, 702, 602], [463, 93, 571, 487]]
#     rois = np.array(rois, dtype=np.float32)
#     print rois
#
#     gid, head_target, _ = transform_head(rois, head_bbs, (5, 5))
#     print gid
#     print head_target
