import torch
import torch.nn.functional as F
from ssim import ssim
from geometrics import flow_warp, inverse_warp, pose2flow, mask_gen



def l1_norm(x):
    x = x.abs()
    return x.mean()


def l2_norm(x):
    x = x.norm(dim=1)
    return x.mean()


def gradient(pred):
    dy = pred[:, :, :-1, :] - pred[:, :, 1:, :]
    dx = pred[:, :, :, -1:] - pred[:, :, :, 1:]
    return dx, dy


def upsample(src, shape):
    return F.interpolate(src, shape, mode="bilinear")


def multi_scale_mask(multi_scale, depth, pose, flow, intrinsics, intrinsics_inv):
    masks = []
    d_t0, d_t1 = depth
    f_forward, f_backward = flow

    for s in range(multi_scale):
        flow_rigid_foward = pose2flow(
            d_t0[s], pose, intrinsics, intrinsics_inv)
        flow_rigid_backward = pose2flow(
            d_t1[s], -pose, intrinsics, intrinsics_inv)

        mask0 = mask_gen(f_forward[s], flow_rigid_foward)
        mask1 = mask_gen(f_backward[s], flow_rigid_backward)
        masks.append((mask0, mask1))
    return masks


class Loss:
    def __init__(self, cfg, train=True):
        self.cfg = cfg
        self.weights=cfg['weights']
        self.loss_dict = dict(
            reprojection_loss=0.,
            flow_consistency=0.,
            flow_smo=0.,
            depth_smo=0.,
            disc=0.,
            depth_loss=0.,
            pose_loss=0.,
            flow_loss=0.,
        )
        self.loss_per_iter = 0.
        self.train = train

    def summerize(self, pred, target):
        self._clean()
        B, _, H, W = target['img_src'].size()
        for scale in range(self.cfg.multi_scale):
            weight=self.weights['multi_scale']
            if self.cfg.use_depth:
                depthmap = pred['depthmap'][scale]
                depthmap = upsample(depthmap, (H, W))
                pose = pred['pose']

                self.loss_dict['reprojection_loss'] += self.loss_reprojection(
                    depthmap, pose, target['img_src'], target['img_tgt'],
                    target['intrinsics'], target['intrinsics_inv'],
                    weight=weight)

                self.loss_dict['depth_smo'] += self.loss_smo(depthmap, target['img_src'],
                    weight=weight)

            if self.cfg.use_flow:
                flowmap = pred['flowmap'][scale]
                flowmap = upsample(flowmap, (H, W))
                self.loss_dict['flow_consistency'] += self.loss_flow_consistency(
                    flowmap, target['img_src'], target['img_tgt'],
                    weight=weight)
                self.loss_dict['depth_smo'] += self.loss_smo(
                    depthmap, target['img_src'],
                    weight=weight)

            if self.cfg.use_discriminator:
                pass

        for k in self.loss_dict.keys():
            self.loss_per_iter += self.loss_dict[k]*self.weights[k]
    # nouvel

    def loss_reconstruction(self, img_tgt, img_warped, mask=None):
        if mask is not None:
            img_tgt = img_tgt*mask
            img_warped = img_warped*mask
        ssim = 1-ssim(img_warped, img_tgt)
        l1 = l1_norm(img_warped-img_tgt)
        return ssim*self.cfg['ssim']+l1*self.cfg['l1']

    def loss_reprojection(self, depth, pose, img_src, img_tgt,
                          intrinsics, intrinsics_inv, mask=None, weight=1.):

        B, _, H, W = depth.size()
        img_tgt_warped = inverse_warp(
            img_src, depth, pose, intrinsics, intrinsics_inv)
        loss = self.loss_reconstruction(img_tgt, img_tgt_warped, mask)*weight
        return loss

    def loss_flow_consistency(self, flow, img_src, img_tgt, weight=1.):
        B, _, H, W = flow.size()
        img_warped = flow_warp(img_src, flow)
        loss = self.loss_reconstruction(img_tgt, img_warped)*weight
        return loss

    def loss_smo(self, tgt, img, weight=1.):
        B, _, H, W = tgt.size()
        loss = 0.

        for scale, t in enumerate(tgt):
            grad_target_x, grad_target_y = gradient(t)
            grad_image_x, grad_image_y = gradient(img)
            grad_image_x = torch.mean(grad_image_x, 1, keepdim=True)
            grad_image_y = torch.mean(grad_image_y, 1, keepdim=True)
            loss_smo_x = torch.exp(-grad_image_x)*grad_target_x
            loss_smo_y = torch.exp(-grad_image_y)*grad_target_y
            loss += (loss_smo_x.mean()+loss_smo_y.mean()) * \
                self.cfg.multi_scale_weights[scale]
        return loss*weight

    # supervised
    def loss_depth(self, gt, pred):
        loss = 0.
        for i, p in enumerate(pred):
            B, _, H, W = p.size()
            gt = torch.nn.functional.adaptive_avg_pool2d(gt, (H, W))
            dist = l2_norm(gt-p)
            loss += dist*multi_scale_weights[i]
        return loss

    def loss_flow(self, gt, pred):
        _, _, h_pred, w_pred = pred.size()
        bs, nc, h_gt, w_gt = gt.size()
        u_gt, v_gt = gt[:, 0, :, :], gt[:, 1, :, :]
        # resize to the gt
        pred = F.upsample(pred, size=(h_gt, w_gt), mode='bilinear')
        # mind the scale
        u_pred = pred[:, 0, :, :] * (w_gt/w_pred)
        v_pred = pred[:, 1, :, :] * (h_gt/h_pred)

        dist = l1_norm(u_gt - u_pred) + l1_norm(v_gt - v_pred)
        return dist

    def loss_pose(self, gt, pred):         # [B,6]
        return l1_norm(gt-pred)

    def _clean(self):
        self.loss_per_iter = 0.0
        for k in self.loss_dict.keys():
            self.loss_dict[k] = 0.0

    # def loss_reconstruction(img_tgt, img_warped, mask=None, apply_valid=False):

        # B, _, H, W = img_warped.size()
        # valid_loss=0
        # # create valid mask of
        # if apply_valid:
        #     valid_mask = 1 - (img_warped == 0).\
        #                 prod(1, keepdim=True).type_as(img_warped)

        #     if mask is not None:
        #         valid_mask = valid_mask*mask
        # # to make sure the valid mask won't be all zeros
        #     img_warped = img_warped*valid_mask
        #     img_tgt = img_tgt*valid_mask

        #     valid_loss = 1 - valid_mask.sum()/valid_mask.nelement()

        # ssim_loss = 1-ssim(img_warped, img_tgt)
        # l2_loss = l2_norm(img_warped-img_tgt)

        # loss = l2_loss*reconstruction_weights[0] +\
        #     ssim_loss*reconstruction_weights[1] +\
        #     valid_loss*reconstruction_weights[2]
        # return loss

    # nouvel
    # 还没实现 edge-aware

    # def loss_smoothness(self,outputs):
        # loss = 0.
        # for i, output in enumerate(outputs):
        #     output = torch.unsqueeze(output,dim=1)
        #     dx, dy = gradient(output)
        #     dx2, dxdy = gradient(dx)
        #     dydx, dy2 = gradient(dy)
        #     loss += ((dx2.abs().mean() +
        #             dxdy.abs().mean() +
        #             dydx.abs().mean() +
        #             dy2.abs().mean()) *
        #             multi_scale_weights[i])
        # return loss

    # nouvel

    # unsupervised loss
    # def loss_flow_consistency(self, flow, img_src, img_tgt, multi_scale=0):

    #     loss = 0.
    #     if multi_scale > 0:
    #         for s in range(multi_scale):
    #             B, _, H, W = flow[s].size()
    #             img_src_s = torch.nn.functional.adaptive_avg_pool2d(
    #                 img_src, (H, W))
    #             img_tgt_s = torch.nn.functional.adaptive_avg_pool2d(
    #                 img_tgt, (H, W))

    #             img_tgt_warped = flow_warp(img_src_s, flow[s])

    #             loss+=(loss_reconstruction(img_tgt_s, img_tgt_warped)*multi_scale_weights[s])
    #     else:
    #         loss += loss_reconstruction(img_tgt, flow_warp(img_src, flow[0]))

    #     return loss

    # nouvel

    # def loss_depth_consistency(
        #     depth_t0, depth_t1, pose, img_src, img_tgt,
        #     intrinsics, intrinsics_inv, mask=None, multi_scale=0):

        # loss = 0

        # if multi_scale > 0:
        #     origine = img_src.size()[-1]
        #     for s in range(multi_scale):
        #         _, H, W = depth_t0[s].size()
        #         ratio = float(origine)/W

        #         img_src_s = torch.nn.functional.adaptive_avg_pool2d(
        #             img_src, (H, W))
        #         img_tgt_s = torch.nn.functional.adaptive_avg_pool2d(
        #             img_tgt, (H, W))

        #         intrinsics_s = torch.cat(
        #             (intrinsics[:, 0:2]/ratio, intrinsics[:, 2:]), dim=1)
        #         intrinsics_inv_s = torch.cat(
        #             (intrinsics_inv[:, 0:2]/ratio, intrinsics[:, 2:]), dim=1)

        #         img_tgt_warped = inverse_warp(
        #             img_src_s, depth_t0[s], pose, intrinsics_s, intrinsics_inv_s)
        #         img_src_warped = inverse_warp(
        #             img_tgt_s, depth_t1[s], -pose, intrinsics_s, intrinsics_inv_s)

        #         if mask is not None:
        #             l = loss_reconstruction(img_tgt_s, img_tgt_warped, mask[s][1]) +\
        #                 loss_reconstruction(img_src_s, img_src_warped, mask[s][0])
        #         else:
        #             l = loss_reconstruction(img_tgt_s, img_tgt_warped) +\
        #                 loss_reconstruction(img_src_s, img_src_warped)

        #         loss += (l*multi_scale_weights[s])

        # else:
        #     img_tgt_warped = inverse_warp(
        #         img_src, depth_t0[0], pose, intrinsics, intrinsics_inv)
        #     img_src_warped = inverse_warp(
        #         img_tgt, depth_t1[0], -pose, intrinsics, intrinsics_inv)

        #     if mask is not None:
        #         l = loss_reconstruction(img_tgt, img_tgt_warped, mask[0][1]) + \
        #             loss_reconstruction(img_src, img_src_warped, mask[0][0])
        #     else:
        #         l = loss_reconstruction(img_tgt, img_tgt_warped) +\
        #             loss_reconstruction(img_src, img_src_warped)

        #     loss += l

        # return loss

    # def loss_sum(loss_dict):
    #     alpha = cfg.loss_weight
    #     loss = 0
    #     for key in loss_dict.keys():
    #         loss += (loss_dict[key]*alpha[key])
    #     return loss
